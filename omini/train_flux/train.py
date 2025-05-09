import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import lightning as L
import os
import time
import yaml

from datasets import load_dataset

from .training_model import (
    BaseModel,
    TrainingCallback,
    get_rank,
    get_config,
    init_wandb,
)
from ..pipeline.flux_omini import Condition, generate


class CustomDataset(Dataset):
    def __getitem__(self, idx):
        # TODO: Implement the logic to load your custom dataset
        raise NotImplementedError("Custom dataset loading not implemented")


class CustomModel(BaseModel):
    @torch.no_grad()
    def generate_a_sample(self, save_path, file_name):
        # TODO: Implement the logic to generate a sample using the model
        raise NotImplementedError("Sample generation not implemented")


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
    dataset = CustomDataset()
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = CustomModel(
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
