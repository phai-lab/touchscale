import os
from typing import Any, Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

# Attention backend for the manual joint prefix+suffix layer loop, chosen ONCE at import.
# "eager" (default) = HF eager_attention_forward — byte-identical to the historical path (the
# gate_c / iron-rule baseline; existing lines keep exactly today's numerics). "sdpa" = fused
# torch.nn.functional.scaled_dot_product_attention: no seq×seq fp32 softmax materialization,
# which is what capped e2e at ~16/GPU (the reference implementation's large-batch ran 128/GPU on a fused path). Same
# math, NOT bitwise-identical — opt in per-run via VTLA_ATTN_IMPL=sdpa, never by default.
_VTLA_ATTN_IMPL = os.environ.get("VTLA_ATTN_IMPL", "eager")


def _joint_attention(module, query_states, key_states, value_states, attention_mask, scaling):
    if _VTLA_ATTN_IMPL != "sdpa":
        return modeling_gemma.eager_attention_forward(
            module, query_states, key_states, value_states, attention_mask, scaling
        )
    key = modeling_gemma.repeat_kv(key_states, module.num_key_value_groups)
    value = modeling_gemma.repeat_kv(value_states, module.num_key_value_groups)
    # eager applies the additive causal mask sliced to the key length; sdpa takes the same slice.
    # F.sdpa requires the additive mask dtype to MATCH query dtype (bf16); the mask arrives fp32
    # (eager tolerates that because it adds in fp32) — cast or it raises "invalid dtype for bias".
    mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
    if mask is not None and mask.dtype != query_states.dtype and mask.dtype != torch.bool:
        mask = mask.to(query_states.dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        query_states, key, value, attn_mask=mask, scale=scaling
    )
    # match eager_attention_forward's output layout: (B, S, H, D)
    return out.transpose(1, 2).contiguous(), None


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _prefix_layer(self, layer_idx, hidden, attention_mask, position_ids):
        layer = self.paligemma.language_model.layers[layer_idx]
        hidden_n, gate = layer.input_layernorm(hidden, cond=None)
        input_shape = hidden_n.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        k = layer.self_attn.k_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
        cos, sin = self.paligemma.model.language_model.rotary_emb(dummy, position_ids)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        scaling = layer.self_attn.scaling
        att, _ = _joint_attention(layer.self_attn, q, k, v, attention_mask, scaling)
        head_dim = layer.self_attn.head_dim
        att = att.reshape(q.shape[0], -1, 1 * 8 * head_dim)
        if att.dtype != layer.self_attn.o_proj.weight.dtype:
            att = att.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(att)
        out = modeling_gemma._gated_residual(hidden, out, gate)  # noqa: SLF001
        after = out.clone()
        out, gate2 = layer.post_attention_layernorm(out, cond=None)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out = out.to(dtype=torch.bfloat16)
        out = layer.mlp(out)
        out = modeling_gemma._gated_residual(after, out, gate2)  # noqa: SLF001
        return out, k, v

    def forward_prefix_cache(self, prefix_embs, attention_mask, position_ids):
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        hidden = prefix_embs
        cached_k, cached_v = [], []
        for layer_idx in range(num_layers):
            if self.training:
                hidden, k, v = torch.utils.checkpoint.checkpoint(
                    self._prefix_layer, layer_idx, hidden, attention_mask, position_ids,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hidden, k, v = self._prefix_layer(layer_idx, hidden, attention_mask, position_ids)
            cached_k.append(k)
            cached_v.append(v)
        vl_ctx, _ = self.paligemma.language_model.norm(hidden, cond=None)
        return vl_ctx, cached_k, cached_v

    def _suffix_layer(self, layer_idx, hidden, attention_mask, position_ids, prefix_k, prefix_v, adarms_cond):
        # Suffix q/k/v from the EXPERT layer; attention (scaling/GQA) uses PaliGemma's self_attn,
        # matching compute_layer_complete's joint attention. Keys/values = [cached prefix ; suffix].
        elayer = self.gemma_expert.model.layers[layer_idx]
        player = self.paligemma.language_model.layers[layer_idx]
        hidden_n, gate = elayer.input_layernorm(hidden, cond=adarms_cond)
        input_shape = hidden_n.shape[:-1]
        hidden_shape = (*input_shape, -1, elayer.self_attn.head_dim)
        q = elayer.self_attn.q_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        k = elayer.self_attn.k_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        v = elayer.self_attn.v_proj(hidden_n).view(hidden_shape).transpose(1, 2)
        dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
        cos, sin = self.paligemma.model.language_model.rotary_emb(dummy, position_ids)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        key_states = torch.cat([prefix_k.to(k.dtype), k], dim=2)
        value_states = torch.cat([prefix_v.to(v.dtype), v], dim=2)
        scaling = player.self_attn.scaling
        att, _ = _joint_attention(player.self_attn, q, key_states, value_states, attention_mask, scaling)
        head_dim = elayer.self_attn.head_dim
        att = att.reshape(q.shape[0], -1, 1 * 8 * head_dim)
        if att.dtype != elayer.self_attn.o_proj.weight.dtype:
            att = att.to(elayer.self_attn.o_proj.weight.dtype)
        out = elayer.self_attn.o_proj(att)
        out = modeling_gemma._gated_residual(hidden, out, gate)  # noqa: SLF001
        after = out.clone()
        out, gate2 = elayer.post_attention_layernorm(out, cond=adarms_cond)
        if elayer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out = out.to(dtype=torch.bfloat16)
        out = elayer.mlp(out)
        out = modeling_gemma._gated_residual(after, out, gate2)  # noqa: SLF001
        return out

    def forward_suffix_cached(self, suffix_embs, attention_mask, position_ids, cached_k, cached_v, adarms_cond):
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        hidden = suffix_embs
        for layer_idx in range(num_layers):
            if self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._suffix_layer, layer_idx, hidden, attention_mask, position_ids,
                    cached_k[layer_idx], cached_v[layer_idx], adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hidden = self._suffix_layer(
                    layer_idx, hidden, attention_mask, position_ids,
                    cached_k[layer_idx], cached_v[layer_idx], adarms_cond,
                )
        suffix_output, _ = self.gemma_expert.model.norm(hidden, cond=adarms_cond)
        return suffix_output

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Any | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it.
            # VTLA_NO_CKPT=1 disables the forcing. Validate changes to this behavior with
            # scripts/verify_prefix_cache.py.
            if (
                os.environ.get("VTLA_NO_CKPT") != "1"
                and self.training
                and hasattr(self.gemma_expert.model, "gradient_checkpointing")
            ):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                att_output, _ = _joint_attention(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
