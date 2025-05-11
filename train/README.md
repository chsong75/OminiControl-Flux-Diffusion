# Training for FLUX


## Table of Contents
- [Training for FLUX](#training-for-flux)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Basic Training](#basic-training)
    - [Tasks from OminiControl](#tasks-from-ominicontrol)
    - [Customize Your Own Task](#customize-your-own-task)
      - [Understanding Position delta](#understanding-position-delta)
    - [Training Configuration](#training-configuration)
      - [Optimizer](#optimizer)
      - [LoRA Configuration](#lora-configuration)
      - [Trainable Modules](#trainable-modules)
  - [Advanced Training](#advanced-training)
    - [Multi-condition](#multi-condition)
    - [Towards Efficient Generation (OminiControl2)](#towards-efficient-generation-ominicontrol2)
      - [Feature Reuse (KV-Cache)](#feature-reuse-kv-cache)
      - [Compact Encoding Representation](#compact-encoding-representation)
      - [Token Integration (for Fill task)](#token-integration-for-fill-task)

## Environment Setup
1. Environment setup:
    ```bash
    conda create -n omini python=3.10
    conda activate omini
    ```
2. Dependency for training:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation
1. Download dataset [Subject200K](https://huggingface.co/datasets/Yuanshi/Subjects200K). (**subject-driven generation**)
    ```
    bash train/script/data_download/data_download1.sh
    ```
2. Download dataset [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M). (**spatial control task**)
    ```
    bash train/script/data_download/data_download2.sh
    ```
    **Note:** By default, only a few files are downloaded. You can modify `data_download2.sh` to download additional datasets. Remember to update the config file to specify the training data accordingly.

## Quick Start
To quickly start training, you can use these demo scripts:

**1. Subject-driven generation**
  ```bash
bash train/script/train_subject.sh
```

**2. Spatial control tasks** (e.g., Canny-to-image, Image colorization, Depth map to image, etc.):
```bash
bash train/script/train_spatial.sh
```
**3. Multi-condition training**
```bash
bash train/script/train_multicondition.sh
```
**4. Feature reuse** ([OminiControl2](https://arxiv.org/abs/2503.08280))
```bash
bash train/script/train_feature_reuse.sh
```

**5. Compact token representation** ([OminiControl2](https://arxiv.org/abs/2503.08280))
```bash
bash train/script/train_compact_token_representation.sh
```

**6. Token integration** ([OminiControl2](https://arxiv.org/abs/2503.08280))
```bash
bash train/script/train_token_intergration.sh
```


## Basic Training
### Tasks from OminiControl
<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/ariXv-2411.15098-A42C25.svg" alt="arXiv"></a>

1. Subject-driven generation
    ```bash
    bash train/script/train_subject.sh
    ```
2. Spatial control tasks (*canny-to-image* as an example)
    ```bash
    bash train/script/train_spatial.sh
    ```
    <details>
    <summary>Supported tasks</summary>

    * Canny edge to image (`canny`)
    * Image colorization (`coloring`)
    * Image deblurring (`deblurring`)
    * Depth map to image (`depth`)
    * Image to depth map (`depth_pred`)
    * Image inpainting (`fill`)
    * Super resolution (`sr`)
   
    🌟 You can modify the `condition_type` parameter in the config file to switch between different tasks.
    </details>

**Note**: Detailed WanDB settings and GPU settings can be found in the **script files** (`train/configs/`) and the **config files** (`train/script/`).

### Customize Your Own Task
You can customize your own task by constructing a new dataset and modifying the testing code based on the template provided in `train_flux/train.py`.

1. **Create a new dataset:**
A custom dataset(`CustomDataset`) should be in the similar format as the `ImageConditionDataset` dataset in `omini/train_flux/train_single.py`. Each sample should contain 3 main components:
    - Image: the target generated image. (`image`)
    - Text: the description of the image. (`description`)
    - Conditions: the image conditions to be used for generation.
    - Position delta 
        1. Use `position_delta = (0, 0)` if you want the condition image to align exactly with the generated image (like Canny-to-image).
        2. Use `position_delta = (0, -a)` if you want the condition image to be separated (e.g., for subject-driven or style transfer tasks). Here, a = condition width / 16.
        > **Why?** 
        > You can imagine the model places both the condition and generated image in a shared spatial grid (like a coordinate system). `position_delta` controls how much to shift the condition image in that space.
        > Each shift unit equals the size of one patch (usually 16 pixels). So for a 512px-wide condition image (which has 32 patches), `position_delta = (0, -32)` moves it fully to the left, avoiding overlap. 
        > This lets you control whether the condition and generated image share the same space or are positioned side-by-side.

2. **Modify the testing code:**
You should define the `test_function()` in `base_model.py`. This function will be called during the testing phase. You can refer to the `test_function()` function in the `train_single.py` for more details. **Don't forget to keep the `position_delta` parameter same as the one in the dataset.**



### Training Configuration
#### Optimizer

This repository uses `Prodigy` as the default optimizer. You can change it to `AdamW` by modifying the `optimizer` parameter in the config file.
```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.001
```

#### LoRA Configuration
The LoRA rank is set to 4 by default. You can increase it if the task is complex. (Note: you need to keep the `r` parameter and `lora_alpha` parameter the same).
```yaml
lora_config:
  r: 128
  lora_alpha: 128
```

#### Trainable Modules

The `target_modules` parameter is a **regex pattern**. Which is used to specify the modules to be trained. You can refer to the [PEFT Doc](https://huggingface.co/docs/peft/package_reference/lora) for more details.


The default training modules includes all the modules which affect the image tokens:
```yaml
target_modules: "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"
```


If you want to train the `to_q` and `to_k` and `to_v` modules, you can set the `target_modules` parameter to:
```yaml
target_modules: "(.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v)"
```

Details effects of the LoRA modules can be found in the [OminiControl paper](https://arxiv.org/abs/2411.15098).




## Advanced Training
### Multi-condition
The naive multi-condition demo is implemented in `train_multicondition.py`. You can run it by:
```bash
bash train/script/train_multicondition.sh
```

### Towards Efficient Generation (OminiControl2)
<a href="https://arxiv.org/abs/2503.08280"><img src="https://img.shields.io/badge/ariXv-2503.08280-A42C25.svg" alt="arXiv"></a>

[OminiControl2](https://arxiv.org/abs/2503.08280) introduces several techniques to improve the efficiency of the model during inference.
#### Feature Reuse (KV-Cache)
1. To enable feature reuse, you need to enable `independent_condition` in the config file during training.
    ```yaml
    model:
      independent_condition: true
    ```
2. During inference, you can set `kv_cache = True` in the `generate` function to accelerate the generation process.

**Demo:**
```bash
bash train/script/train_feature_reuse.sh
```
**Note:** While feature reuse can significantly speed up the generation process, it may lead to a slight decrease in performance and a increase in training time.


#### Compact Encoding Representation
To use the compact encoding representation, you can first reduce the resolution of the condition image. Then use `position_scale` to align the condition image with the generated image in different resolutions.

For example, the original condition image is 512x512, and the generated image is 256x256. You can set `position_scale = 2` to align the condition image with the generated image.

```diff
train:
  dataset:
    condition_size: 
-     - 512
-     - 512
+     - 256
+     - 256
+   position_scale: 2
    target_size: 
      - 512
      - 512
```
**Demo:**
```bash
bash train/script/train_compact_token_representation.sh
```

#### Token Integration (for Fill task)
To further reduce the number of tokens, you can use the token integration method for the fill task. The token integration method will merge the condition tokens and the generated tokens into a unified token sequence. This will reduce the number of tokens by half.

**Demo:**
```bash
bash train/script/train_token_intergration.sh
```

## Citation
If you find this code useful, please consider citing our paper:
```
@article{tan2024ominicontrol,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Tan, Zhenxiong and Liu, Songhua and Yang, Xingyi and Xue, Qiaochu and Wang, Xinchao},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}

@article{tan2025ominicontrol2,
  title={OminiControl2: Efficient Conditioning for Diffusion Transformers},
  author={Tan, Zhenxiong and Xue, Qiaochu and Yang, Xingyi and Liu, Songhua and Wang, Xinchao},
  journal={arXiv preprint arXiv:2503.08280},
  year={2025}
}
```
