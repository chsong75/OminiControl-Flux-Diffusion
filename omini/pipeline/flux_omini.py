import torch
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    FluxTransformer2DModel,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb

from accelerate.utils import is_torch_version

from .tools import clip_hidden_states


def attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],
    hidden_states2: Optional[List[torch.FloatTensor]] = [],
    position_embs: Optional[List[torch.Tensor]] = None,
) -> torch.FloatTensor:
    bs, _, _ = hidden_states[0].shape

    queries, keys, values = [], [], []

    # Prepare query, key, value for each encoder hidden state (text branch)
    for hidden_state in hidden_states2:
        query = attn.add_q_proj(hidden_state)
        key = attn.add_k_proj(hidden_state)
        value = attn.add_v_proj(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Prepare query, key, value for each hidden state (image branch)
    for hidden_state in hidden_states:
        query = attn.to_q(hidden_state)
        key = attn.to_k(hidden_state)
        value = attn.to_v(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Apply rotary embedding
    if position_embs is not None:
        for i, (query, key) in enumerate(zip(queries, keys)):
            queries[i] = apply_rotary_emb(query, position_embs[i])
            keys[i] = apply_rotary_emb(key, position_embs[i])

    # Attention computation
    attn_output = F.scaled_dot_product_attention(
        torch.cat(queries, dim=2), torch.cat(keys, dim=2), torch.cat(values, dim=2)
    ).to(query.dtype)
    attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)

    # Reshape attention output to match the original hidden states
    offset = 0
    for i, hidden_state in enumerate(hidden_states2):
        hidden_states2[i] = attn_output[:, offset : offset + hidden_state.shape[1]]
        hidden_states2[i] = attn.to_add_out(hidden_states2[i])
        offset += hidden_state.shape[1]

    for i, hidden_state in enumerate(hidden_states):
        hidden_states[i] = attn_output[:, offset : offset + hidden_state.shape[1]]
        if hasattr(attn, "to_out"):
            hidden_states[i] = attn.to_out[0](hidden_states[i])
            hidden_states[i] = attn.to_out[1](hidden_states[i])
        offset += hidden_state.shape[1]

    return (hidden_states, hidden_states2) if hidden_states2 else hidden_states


def block_forward(
    self,
    image_hidden_states: List[torch.FloatTensor],
    text_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    position_embs=None,
):
    txt_n = len(text_hidden_states)

    img_variables = [[None for _ in range(5)] for _ in image_hidden_states]
    txt_variables = [[None for _ in range(5)] for _ in text_hidden_states]

    for i, text_h in enumerate(text_hidden_states):
        txt_variables[i] = self.norm1_context(text_h, emb=tembs[i])

    for i, image_h in enumerate(image_hidden_states):
        img_variables[i] = self.norm1(image_h, emb=tembs[i + txt_n])

    # Attention.
    img_attn_output, txt_attn_output = attn_forward(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each[0] for each in txt_variables],
        position_embs=position_embs,
    )

    for i in range(len(text_hidden_states)):
        norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
        text_hidden_states[i] += txt_attn_output[i] * gate_msa.unsqueeze(1)
        norm_h = (
            self.norm2_context(text_hidden_states[i]) * (1 + scale_mlp[:, None])
            + shift_mlp[:, None]
        )
        text_hidden_states[i] += self.ff_context(norm_h) * gate_mlp.unsqueeze(1)
        text_hidden_states[i] = clip_hidden_states(text_hidden_states[i])

    for i in range(len(image_hidden_states)):
        norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_hidden_states[i] += img_attn_output[i] * gate_msa.unsqueeze(1)
        norm_h = (
            self.norm2(image_hidden_states[i]) * (1 + scale_mlp[:, None])
            + shift_mlp[:, None]
        )
        image_hidden_states[i] += self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_hidden_states[i] = clip_hidden_states(image_hidden_states[i])

    return image_hidden_states, text_hidden_states


def single_block_forward(
    self,
    hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    position_embs=None,
):
    residual = [h for h in hidden_states]
    mlp_hidden_states, gates = [[None for _ in hidden_states] for _ in range(2)]

    for i, hidden_state in enumerate(hidden_states):
        # [NOTE]!: This function's output is slightly DIFFERENT from the original
        # FLUX version. In the original implementation, the gates were computed using
        # the combined hidden states from both the image and text branches. Here, each
        # branch computes its gate using only its own hidden state.
        hidden_states[i], gates[i] = self.norm(hidden_state, emb=tembs[i])
        mlp_hidden_states[i] = self.act_mlp(self.proj_mlp(hidden_states[i]))

    attn_outputs = attn_forward(self.attn, hidden_states, position_embs=position_embs)

    for i in range(len(hidden_states)):
        h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
        hidden_states[i] = gates[i].unsqueeze(1) * self.proj_out(h) + residual[i]
        hidden_states[i] = clip_hidden_states(hidden_states[i])

    return hidden_states


def tranformer_forward(
    transformer: FluxTransformer2DModel,
    image_features: List[torch.Tensor],
    text_features: List[torch.Tensor] = None,
    img_ids: List[torch.Tensor] = None,
    txt_ids: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    guidances: List[torch.Tensor] = None,
    **kwargs: dict,
):
    self = transformer
    txt_n = len(text_features) if text_features is not None else 0

    # Preprocess the image_features
    image_hidden_states = []
    for image_feature in image_features:
        image_hidden_states.append(self.x_embedder(image_feature))

    # Preprocess the text_features
    text_hidden_states = []
    for text_feature in text_features:
        text_hidden_states.append(self.context_embedder(text_feature))

    # Prepare embeddings of (timestep, guidance, pooled_projections)
    assert len(timesteps) == len(image_features) + len(text_features)

    def get_temb(timestep, guidance, pooled_projection):
        if guidance is not None:
            guidance = guidance.to(image_hidden_states[0].dtype) * 1000
            return self.time_text_embed(timestep, guidance, pooled_projection)
        else:
            return self.time_text_embed(timestep, pooled_projection)

    tembs = [get_temb(*each) for each in zip(timesteps, guidances, pooled_projections)]

    # Prepare position embeddings for each token
    position_embs = [self.pos_embed(each) for each in (*txt_ids, *img_ids)]

    # Prepare the gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    # dual branch blocks
    for block in self.transformer_blocks:
        kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states, text_hidden_states = block_forward(**kwargs)

    # combine image and text hidden states then pass through the single transformer blocks
    all_hidden_states = [*text_hidden_states, *image_hidden_states]
    for block in self.single_transformer_blocks:
        kwargs = {
            "self": block,
            "hidden_states": all_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
        }
        if self.training and self.gradient_checkpointing:
            all_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward, **kwargs, **gckpt_kwargs
            )
        else:
            all_hidden_states = single_block_forward(**kwargs)

    image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
    output = self.proj_out(image_hidden_states)

    return (output,)


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Prepare prompt embeddings
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            noise_pred = tranformer_forward(
                self.transformer,
                image_features=[latents],
                text_features=[prompt_embeds],
                img_ids=[latent_image_ids],
                txt_ids=[text_ids],
                timesteps=[timestep, timestep],
                pooled_projections=[pooled_prompt_embeds, pooled_prompt_embeds],
                guidances=[guidance, guidance],
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
