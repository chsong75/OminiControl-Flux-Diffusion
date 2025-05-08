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
from ..pipeline.flux_omini import Condition, convert_to_condition, generate
from .train_base import ImageConditionDataset


class ImageMultiConditionDataset(ImageConditionDataset):
    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_imgs, position_deltas = [], []
        for c_type in self.condition_type:
            condition_img, position_delta = self.__get_condition__(image, c_type)
            condition_imgs.append(condition_img.convert("RGB"))
            position_deltas.append(position_delta)

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_imgs = [
                Image.new("RGB", condition_size)
                for _ in range(len(self.condition_type))
            ]

        return_dict = {
            "image": self.to_tensor(image),
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }

        for i, c_type in enumerate(self.condition_type):
            return_dict[f"condition_{i}"] = self.to_tensor(condition_imgs[i])
            return_dict[f"condition_type_{i}"] = self.condition_type[i]
            return_dict[f"position_delta_{i}"] = position_deltas[i]
            return_dict[f"position_scale_{i}"] = position_scale

        return return_dict


class OminiModel(BaseModel):
    @torch.no_grad()
    def generate_a_sample(self, save_path, file_name):
        condition_size = self.training_config["dataset"]["condition_size"]
        target_size = self.training_config["dataset"]["target_size"]

        position_delta = self.training_config["dataset"].get("position_delta", [0, 0])
        position_scale = self.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        condition_type = self.training_config["condition_type"]
        test_list = []

        condition_list = []
        for i, c_type in enumerate(condition_type):
            if c_type in ["canny", "coloring", "deblurring", "depth"]:
                image = Image.open("assets/vase_hq.jpg")
                image = image.resize(condition_size)
                condition_img = convert_to_condition(c_type, image, 5)
            elif c_type == "fill":
                condition_img = image.resize(condition_size).convert("RGB")
                w, h = image.size
                x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
                y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                if random.random() > 0.5:
                    mask = Image.eval(mask, lambda a: 255 - a)
                condition_img = Image.composite(
                    image, Image.new("RGB", image.size, (0, 0, 0)), mask
                )
            else:
                raise NotImplementedError
            condition = Condition(
                condition_img,
                self.adapter_names[i + 2],
                position_delta,
                position_scale,
            )
            condition_list.append(condition)
        test_list.append((condition_list, "A beautiful vase on a table."))
        os.makedirs(save_path, exist_ok=True)
        for i, (condition, prompt) in enumerate(test_list):
            res = generate(
                self.flux_pipe,
                prompt=prompt,
                conditions=condition_list,
                height=target_size[0],
                width=target_size[1],
                generator=generator,
                model_config=self.model_config,
                kv_cache=self.model_config.get("independent_condition", False),
            )
            file_path = os.path.join(
                save_path, f"{file_name}_{'|'.join(condition_type)}_{i}.jpg"
            )
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
        dataset = ImageMultiConditionDataset(
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

    cond_n = len(training_config["condition_type"])

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, *["default"] * cond_n],
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
