import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict

from typing import List

import prodigyopt


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class BaseModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = ["default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config, adapter_names)

        self.to(device).to(dtype)

    def init_lora(
        self, lora_path: str, lora_config: dict, adapter_names: List[str] = ["default"]
    ):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            for adapter_name in adapter_names:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str, adapter_names: List[str] = ["default"]):
        for adapter_name in adapter_names:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step not implemented.")

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            pl_module.generate_a_sample(
                f"{self.save_path}/{self.run_name}/output", f"lora_{self.total_steps}"
            )
            pl_module.train()
