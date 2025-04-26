import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import lightning as L
import os
import time
import random
import yaml
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_dataset

from .model import BaseModel, TrainingCallback, get_rank, get_config, init_wandb
from ..pipeline.flux_omini import Condition, convert_to_condition, generate


class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": np.array(
                [0, -self.condition_size // 16]
            ),  # 16 is the downscale factor of the image
            "description": description,
            **({"pil_image": image} if self.return_pil_image else {}),
        }


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_condition__(self, image, condition_type):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        if condition_type in ["canny", "coloring", "deblurring", "depth"]:
            condition_img = image.resize((condition_size, condition_size))
            kwargs = {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize((condition_size, condition_size))
            image = depth_img.resize((condition_size, condition_size))
        elif condition_type == "fill":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
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
        elif condition_type == "sr":
            condition_img = image.resize((condition_size, condition_size))
            position_delta = np.array([0, -condition_size // 16])
        else:
            raise ValueError(f"Condition type {condition_type} is not  implemented.")
        return condition_img, position_delta

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_img, position_delta = self.__get_condition__(
            image, self.condition_type
        )

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (condition_size, condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }


class OminiModel(BaseModel):
    @torch.no_grad()
    def generate_a_sample(self, save_path, file_name):
        # TODO: change this two variables to parameters
        condition_size = self.training_config["dataset"]["condition_size"]
        target_size = self.training_config["dataset"]["target_size"]

        position_delta = self.training_config["dataset"].get("position_delta", [0, 0])
        position_scale = self.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        adapter = self.adapter_names[2]
        condition_type = self.training_config["condition_type"]
        test_list = []

        if condition_type == "subject":
            # Test case1
            image = Image.open("assets/test_in.jpg")
            image = image.resize((condition_size, condition_size))
            prompt = "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene."
            condition = Condition(image, adapter, [0, -32], position_scale)
            test_list.append((condition, prompt))
            # Test case2
            image = Image.open("assets/test_out.jpg")
            image = image.resize((condition_size, condition_size))
            prompt = "In a bright room. It is placed on a table."
            condition = Condition(image, adapter, [0, -32], position_scale)
            test_list.append((condition, prompt))
        elif condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image = Image.open("assets/vase_hq.jpg")
            image = image.resize((condition_size, condition_size))
            condition_img = convert_to_condition(condition_type, image, 5)
            condition = Condition(
                condition_img, adapter, position_delta, position_scale
            )
            test_list.append((condition, "A beautiful vase on a table."))
        elif condition_type == "depth_pred":
            image = Image.open("assets/vase_hq.jpg")
            image = image.resize((condition_size, condition_size))
            condition = Condition(image, adapter, position_delta, position_scale)
            test_list.append((condition, "A beautiful vase on a table."))
        elif condition_type == "fill":
            condition_img = (
                Image.open("./assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            mask = Image.new("L", condition_img.size, 0)
            draw = ImageDraw.Draw(mask)
            a = condition_img.size[0] // 4
            b = a * 3
            draw.rectangle([a, a, b, b], fill=255)
            condition_img = Image.composite(
                condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
            )
            condition = Condition(condition, adapter, position_delta, position_scale)
            test_list.append((condition, "A beautiful vase on a table."))
        elif condition_type == "super_resolution":
            image = Image.open("assets/vase_hq.jpg")
            image = image.resize((condition_size, condition_size))
            condition = Condition(image, adapter, position_delta, position_scale)
            test_list.append((condition, "A beautiful vase on a table."))
        else:
            raise NotImplementedError
        os.makedirs(save_path, exist_ok=True)
        for i, (condition, prompt) in enumerate(test_list):
            res = generate(
                self.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_size,
                width=target_size,
                generator=generator,
                model_config=self.model_config,
                kv_cache=self.model_config.get("independent_condition", False),
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
    if training_config["dataset"]["type"] == "subject":
        dataset = load_dataset("Yuanshi/Subjects200K")

        # Define filter function
        def filter_func(item):
            if not item.get("quality_assessment"):
                return False
            return all(
                item["quality_assessment"].get(key, 0) >= 5
                for key in ["compositeStructure", "objectConsistency", "imageQuality"]
            )

        # Filter dataset
        if not os.path.exists("./cache/dataset"):
            os.makedirs("./cache/dataset")
        data_valid = dataset["train"].filter(
            filter_func,
            num_proc=16,
            cache_file_name="./cache/dataset/data_valid.arrow",
        )
        dataset = Subject200KDataset(
            data_valid,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            padding=training_config["dataset"]["padding"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
        )
    elif training_config["dataset"]["type"] == "img":
        # Load dataset text-to-image-2M
        dataset = load_dataset(
            "webdataset",
            data_files={"train": training_config["dataset"]["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )
        dataset = ImageConditionDataset(
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
