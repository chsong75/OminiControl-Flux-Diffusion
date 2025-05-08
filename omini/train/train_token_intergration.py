import torch
from torch.utils.data import DataLoader
import lightning as L
import os
import time
import random
import yaml

from PIL import Image, ImageDraw

from datasets import load_dataset

from .model import BaseModel, TrainingCallback, get_rank, get_config, init_wandb
from ..pipeline.flux_omini import Condition, generate
from .train_base import ImageConditionDataset


class TokenIntergrationDataset(ImageConditionDataset):
    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        assert self.condition_type == "token_intergration"
        assert (
            image.size[0] % 16 == 0 and image.size[1] % 16 == 0
        ), "Condition size must be divisible by 16"

        # Randomly drop text or image (for training)
        description = "" if random.random() < self.drop_text_prob else description

        # Generate a latent mask
        w, h = image.size[0] // 16, image.size[1] // 16
        while True:
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            is_zero = x1 == x2 or y1 == y2
            is_full = x1 == 0 and y1 == 0 and x2 == w and y2 == h
            if not (is_zero or is_full):
                break
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        if random.random() > 0.5:
            mask = Image.eval(mask, lambda a: 255 - a)
        mask = self.to_tensor(mask).to(bool).reshape(-1)

        return {
            "image": self.to_tensor(image),
            "image_latent_mask": torch.logical_not(mask),
            "condition_0": self.to_tensor(image),
            "condition_type_0": self.condition_type,
            "condition_latent_mask_0": mask,
            "description": description,
        }


class OminiModel(BaseModel):
    @torch.no_grad()
    def generate_a_sample(self, save_path, file_name):
        target_size = self.training_config["dataset"]["target_size"]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        condition_type = self.training_config["condition_type"]
        test_list = []

        # Generate two masks to test inpainting and outpainting.
        mask1 = torch.ones((32, 32), dtype=bool)
        mask1[8:24, 8:24] = False
        mask2 = torch.logical_not(mask1)

        image = Image.open("assets/vase_hq.jpg").resize(target_size)
        condition1 = Condition(
            image, self.adapter_names[2], latent_mask=mask1, is_complement=True
        )
        condition2 = Condition(
            image, self.adapter_names[2], latent_mask=mask2, is_complement=True
        )
        test_list.append((condition1, "A beautiful vase on a table.", mask2))
        test_list.append((condition2, "A beautiful vase on a table.", mask1))

        os.makedirs(save_path, exist_ok=True)
        for i, (condition, prompt, latent_mask) in enumerate(test_list):
            res = generate(
                self.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_size[0],
                width=target_size[1],
                generator=generator,
                model_config=self.model_config,
                kv_cache=self.model_config.get("independent_condition", False),
                latent_mask=latent_mask,
            )
            file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            res.images[0].save(file_path)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "img":
        # Load dataset text-to-image-2M
        dataset = load_dataset(
            "webdataset",
            data_files={"train": training_config["dataset"]["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )
        dataset = TokenIntergrationDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )
    else:
        raise NotImplementedError

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, "default"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        training_callbacks = [TrainingCallback(run_name, training_config)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
