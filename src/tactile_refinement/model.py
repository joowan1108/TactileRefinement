"""Frozen SmolVLA with tactile-conditioned FFN LoRA and full-chunk flow handoff."""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from lerobot.policies.smolvla.modeling_smolvla import (
    make_att_2d_masks, create_sinusoidal_pos_embedding,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope

from .representation import (TactileRepresentation, TokenDecoder, Predictor, CoarseChangeHead,
                             SpatialTactileResampler)
from .losses import (reconstruction_loss, latent_loss, contrastive_loss,
                     refinement_bridge, velocity_loss, n0_grounding_loss)


@dataclass
class Prefix:
    features: torch.Tensor
    visual_language_mask: torch.Tensor
    pad_mask: torch.Tensor
    kv: dict
    state: torch.Tensor
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


@torch.no_grad()
def encode_prefix(baseline, batch):
    """Current RGB/text/state only. No tactile or future data accepted here."""
    images, masks = baseline.prepare_images(batch)
    state = baseline.prepare_state(batch)
    core = baseline.model
    embeddings, pad, att = core.embed_prefix(
        images, masks, batch['observation.language.tokens'],
        batch['observation.language.attention_mask'], state=state)
    (features, _), kv = core.vlm_with_expert(
        attention_mask=make_att_2d_masks(pad, att), position_ids=pad.cumsum(1)-1,
        inputs_embeds=[embeddings, None], use_cache=True, fill_kv_cache=True)
    # State occupies its own later block: the predictor receives VL features only.
    return Prefix(features.detach(), pad.bool() & (att == 0), pad,
                  {i: {k: v.detach() for k, v in layer.items()} for i, layer in kv.items()},
                  state.detach(), embeddings.detach(), att)


@torch.no_grad()
def refresh_prefix_state(baseline, prefix, batch):
    """Reuse VL features/KV and recompute only the current proprioception token.

    SmolVLA's block mask prevents VL tokens attending to state. Their cached
    features and keys/values therefore remain valid when state changes. Replace
    the state slot at every VLM layer, preserving RoPE positions and padding;
    updating just Prefix.state would leave the baseline handoff's KV stale.
    """
    state_slots = prefix.attention_mask.bool()
    if not torch.all(state_slots.sum(1) == 1):
        raise ValueError('Cached VL inference requires exactly one state token')
    indices = state_slots.long().argmax(1)
    if not torch.all(indices == indices[0]):
        raise ValueError('State token positions must agree across the batch')
    index = int(indices[0])
    if not torch.all(prefix.pad_mask[:, index]):
        raise ValueError('State token cannot be padded')
    state = baseline.prepare_state(batch)
    if state.shape != prefix.state.shape:
        raise ValueError('Cached prefix and current state batch shapes differ')
    core = baseline.model
    wrapper = core.vlm_with_expert
    text_model = wrapper.get_vlm_model().text_model
    hidden = core.state_proj(state)[:, None, :]
    embeddings = prefix.embeddings.clone()
    embeddings[:, index:index+1] = hidden.to(embeddings.dtype)
    # Match the full prefix's concatenation dtype before entering the VLM.
    hidden = hidden.to(prefix.embeddings.dtype)
    positions = (prefix.pad_mask.cumsum(1)-1)[:, index:index+1]
    mask = make_att_2d_masks(prefix.pad_mask, prefix.attention_mask)[:, index:index+1]
    cache = {}
    for layer_index in range(wrapper.num_vlm_layers):
        layer = text_model.layers[layer_index]
        normalized = layer.input_layernorm(hidden).to(layer.self_attn.q_proj.weight.dtype)
        shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
        query = apply_rope(layer.self_attn.q_proj(normalized).view(shape), positions)
        key = apply_rope(layer.self_attn.k_proj(normalized).view(shape), positions)
        value = layer.self_attn.v_proj(normalized).view(shape)
        updated = {name: tensor.clone() for name, tensor in prefix.kv[layer_index].items()}
        updated['key_states'][:, index:index+1] = key
        updated['value_states'][:, index:index+1] = value
        cache[layer_index] = updated
        attention = wrapper.get_attention_interface()(
            mask, len(state), layer.self_attn.head_dim, query,
            updated['key_states'], updated['value_states'])
        # Match upstream's in-place residual dtype behavior under bfloat16
        # autocast as well as its mathematical operations.
        residual = layer.self_attn.o_proj(attention.to(layer.self_attn.o_proj.weight.dtype))
        residual += hidden
        hidden = layer.mlp(layer.post_attention_layernorm(residual))
        hidden += residual
    features = prefix.features.clone()
    features[:, index:index+1] = text_model.norm(hidden).to(features.dtype)
    return replace(prefix, state=state.detach(), embeddings=embeddings.detach(),
                   features=features.detach(), kv=cache)


class TactileResidual(nn.Module):
    def __init__(self, hidden, latent_width):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, 8, kdim=latent_width, vdim=latent_width,
                                                dropout=0.0, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x, predicted):
        update, _ = self.attention(self.norm(x), predicted, predicted, need_weights=False)
        return x+update


class LowRankDelta(nn.Module):
    """Unmerged LoRA delta; the original linear layer lives only in the baseline."""

    def __init__(self, linear, rank, alpha):
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA requires positive rank and finite positive alpha')
        self.scaling = alpha / rank
        self.a = nn.Linear(linear.in_features, rank, bias=False)
        self.b = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x)) * self.scaling


class CachedTactileExpert(nn.Module):
    """Tactile refinement route over the SAME frozen baseline action expert.

    SmolVLA manually invokes attention/MLP projections instead of layer.forward;
    this loop adds FFN LoRA and tactile residuals without modifying its baseline
    forward. Legacy copied experts are supported only for old configurations.
    """
    forward_attn_layer = SmolVLMWithExpertModel.forward_attn_layer
    forward_cross_attn_layer = SmolVLMWithExpertModel.forward_cross_attn_layer
    eager_attention_forward = SmolVLMWithExpertModel.eager_attention_forward
    get_attention_interface = SmolVLMWithExpertModel.get_attention_interface

    def __init__(self, core, latent_width=256, *, implementation='shared_ffn_lora_v1',
                 lora_rank=32, lora_alpha=32):
        super().__init__()
        if implementation not in ('shared_ffn_lora_v1', 'copied_expert_v1'):
            raise ValueError(f'Unknown refinement implementation: {implementation}')
        self.implementation = implementation
        source = core.vlm_with_expert
        self.config = deepcopy(core.config)
        shared = implementation == 'shared_ffn_lora_v1'
        # Non-registered aliases: baseline owns these modules and handles device,
        # dtype and eval mode. Adapter .train()/requires_grad_()/state_dict() must
        # never recurse into shared original weights or serialize them twice.
        if shared:
            object.__setattr__(self, 'lm_expert', source.lm_expert)
        else:
            self.lm_expert = deepcopy(source.lm_expert)
        for name in ('action_in_proj', 'action_out_proj', 'action_time_mlp_in',
                     'action_time_mlp_out', 'state_proj'):
            if shared:
                object.__setattr__(self, name, getattr(core, name))
            else:
                setattr(self, name, deepcopy(getattr(core, name)))
        for name in ('num_vlm_layers', 'num_expert_layers', 'self_attn_every_n_layers',
                     'attention_mode', 'expert_hidden_size', 'num_attention_heads', 'num_key_value_heads'):
            setattr(self, name, getattr(source, name))
        self.head_dim = source.config.text_config.head_dim
        if shared:
            self.ffn_lora = nn.ModuleList([
                nn.ModuleDict({name: LowRankDelta(getattr(layer.mlp, name), lora_rank, lora_alpha)
                               for name in ('gate_proj', 'up_proj', 'down_proj')})
                for layer in self.lm_expert.layers])
        self.tactile = nn.ModuleList([TactileResidual(self.expert_hidden_size, latent_width)
                                      for _ in self.lm_expert.layers])
        # Retain initial baseline behavior while opening a trainable state route.
        self.state_adapter = nn.Linear(self.state_proj.out_features, self.expert_hidden_size)
        nn.init.zeros_(self.state_adapter.weight)
        nn.init.zeros_(self.state_adapter.bias)
        # Distinguish shared LoRA checkpoints from legacy copied-expert weights.
        self.register_buffer('smolvla_initialization_version', torch.tensor(2 if shared else 1))

    def feed_forward(self, index, x):
        mlp = self.lm_expert.layers[index].mlp
        if self.implementation == 'copied_expert_v1':
            return mlp(x)
        delta = self.ffn_lora[index]
        gated = mlp.act_fn(mlp.gate_proj(x) + delta['gate_proj'](x))
        intermediate = gated * (mlp.up_proj(x) + delta['up_proj'](x))
        return mlp.down_proj(intermediate) + delta['down_proj'](intermediate)

    def embed_suffix(self, x, tau):
        action = self.action_in_proj(x)
        time = create_sinusoidal_pos_embedding(tau, self.expert_hidden_size,
            self.config.min_period, self.config.max_period, device=x.device).to(action.dtype)
        fused = torch.cat((action, time[:, None].expand_as(action)), -1)
        return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(fused)))

    def forward(self, x, tau, prefix, predicted, *, mask_prefix=False):
        hidden = self.embed_suffix(x, tau)
        hidden = hidden + self.state_adapter(self.state_proj(prefix.state))[:, None]
        b, length = hidden.shape[:2]
        suffix_pad = torch.ones(b, length, dtype=torch.bool, device=x.device)
        # Pinned SmolVLA's action suffix has causal attention across action slots.
        suffix_mask = make_att_2d_masks(suffix_pad, torch.ones_like(suffix_pad))
        if mask_prefix:
            # Remove ALL cached tokens, including the VL-contextualized state
            # token. Raw proprioception still enters through the direct adapter.
            cache = {i: {k: v[:, :0] for k, v in layer.items()} for i, layer in prefix.kv.items()}
            attention_mask = suffix_mask
        else:
            cache = prefix.kv
            attention_mask = torch.cat((prefix.pad_mask[:, None].expand(-1, length, -1), suffix_mask), -1)
        # Keep the original rotary positions in both modes.
        positions = prefix.pad_mask.sum(1)[:, None]+suffix_pad.cumsum(1)-1
        multiple = self.num_vlm_layers // self.num_expert_layers
        mapped = [i//multiple if i == 0 or i % multiple == 0 else None
                  for i in range(self.num_vlm_layers)]
        layers = [[None]*self.num_vlm_layers,
                  [self.lm_expert.layers[i] if i is not None else None for i in mapped]]
        for index, expert_index in enumerate(mapped):
            if expert_index is None:
                continue
            cross_only = (
                'cross' in self.attention_mode and not (
                    self.self_attn_every_n_layers > 0 and index % self.self_attn_every_n_layers == 0)
            )
            if mask_prefix and cross_only:
                # Upstream eager softmax would turn an all-masked finite row
                # into a uniform distribution. Explicitly suppress this branch;
                # retain its MLP, action self-attention layers, and tactile path.
                attention = hidden.new_zeros(b, length, self.num_attention_heads*self.head_dim)
            else:
                method = self.forward_cross_attn_layer if cross_only else self.forward_attn_layer
                outputs, _ = method(layers, [None, hidden], index, positions, attention_mask,
                                    b, self.head_dim, use_cache=True, fill_kv_cache=False,
                                    past_key_values=cache)
                attention = outputs[1] if len(outputs) > 1 else outputs[0]
            layer = self.lm_expert.layers[expert_index]
            residual = hidden + layer.self_attn.o_proj(attention.to(layer.self_attn.o_proj.weight.dtype))
            hidden = residual + self.feed_forward(expert_index, layer.post_attention_layernorm(residual))
            hidden = self.tactile[expert_index](hidden, predicted)
        return self.action_out_proj(self.lm_expert.norm(hidden).float())


class RefinementModel(nn.Module):
    def __init__(self, baseline, width=256, *, stage2_mask_prefix=True, trex_n0_full_grid=False,
                 expert_implementation='shared_ffn_lora_v1', lora_rank=32, lora_alpha=32,
                 resampler_tokens_per_finger=0):
        super().__init__()
        if getattr(baseline.config, 'use_tactile', False) or baseline.config.adapt_to_pi_aloha:
            raise ValueError('Expected the tactile-free UniVTAC joint-position baseline')
        if baseline.config.chunk_size != 50 or not baseline.config.use_cache:
            raise ValueError('Expected 50-action SmolVLA with prefix caching')
        self.baseline = baseline.requires_grad_(False).eval()
        self.trex_n0_full_grid = trex_n0_full_grid
        self.representation = TactileRepresentation(width, full_grid=trex_n0_full_grid)
        self.decoder = CoarseChangeHead(width) if trex_n0_full_grid else TokenDecoder(width)
        self.predictor = Predictor(baseline.model.state_proj.out_features, width,
                                   positions_per_finger=300 if trex_n0_full_grid else 16)
        self.expert = CachedTactileExpert(baseline.model, width, implementation=expert_implementation,
                                         lora_rank=lora_rank, lora_alpha=lora_alpha)
        if resampler_tokens_per_finger and not trex_n0_full_grid:
            raise ValueError('Post-predictor resampling requires the full-grid recipe')
        self.resampler = (SpatialTactileResampler(width, resampler_tokens_per_finger)
                          if resampler_tokens_per_finger else nn.Identity())
        self.stage2_mask_prefix = stage2_mask_prefix
        self.stage = 'grounding' if trex_n0_full_grid else 'encoder'
        self.set_stage(self.stage)

    def set_stage(self, stage):
        if stage not in ('encoder', 'grounding', 'expert', 'joint'):
            raise ValueError(stage)
        if self.trex_n0_full_grid and stage == 'encoder':
            raise ValueError('Pretrained T-Rex backbone is frozen; start with grounding')
        self.stage = stage
        self.requires_grad_(False)
        groups = {'encoder': (self.representation, self.decoder),
                  'grounding': (self.predictor, self.decoder),
                  'expert': (self.expert, self.resampler), 'joint': (self.predictor, self.expert, self.resampler)}
        if self.trex_n0_full_grid:
            groups['grounding'] = (self.representation.projector, self.predictor, self.decoder)
            groups['joint'] = (self.representation.projector, self.predictor, self.expert, self.resampler)
        for module in groups[stage]:
            module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        for module in (self.representation, self.decoder, self.predictor, self.expert, self.resampler):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        if self.trex_n0_full_grid:
            self.representation.encoder.eval()
        return self

    @torch.no_grad()
    def handoff(self, prefix, noise=None):
        core = self.baseline.model
        x = core.sample_noise((len(prefix.state), 50, core.config.max_action_dim), prefix.state.device) if noise is None else noise
        for step in range(6):
            tau = torch.full((len(x),), 1-step*0.1, device=x.device)
            x = x-0.1*core.denoise_step(prefix.pad_mask, prefix.kv, x, tau)
        return x.detach()

    def predicted_tokens(self, depth, prefix):
        if self.trex_n0_full_grid:
            current = self.representation(depth)
        else:
            with torch.no_grad():
                current = self.representation(depth)
        return self.predictor(current, prefix.features, prefix.visual_language_mask)

    def loss(self, batch, tactile, *, lambda_rec=1.0, lambda_nce=0.1):
        if self.stage == 'encoder':
            maps = torch.cat((tactile['depth'], tactile['change']))
            mask = torch.cat((tactile['depth_mask'], tactile['change_mask']))
            # Invalid endpoints contribute neither loss nor encoder BN statistics.
            keep = mask.flatten(1).any(1)
            maps, mask = maps[keep], mask[keep]
            loss = reconstruction_loss(self.decoder(self.representation(maps)), maps, mask)
            return loss, {'reconstruction': loss.detach()}
        prefix = encode_prefix(self.baseline, batch)
        pred = self.predicted_tokens(tactile['depth'], prefix)
        if self.stage == 'grounding':
            with torch.no_grad():
                target = self.representation(tactile['change'])
            valid = tactile['future_valid'] & tactile['change_mask'].flatten(1).any(1)
            if self.trex_n0_full_grid:
                return n0_grounding_loss(pred, target, self.decoder(pred), tactile['change'],
                                         tactile['change_mask'], valid, lambda_rec=lambda_rec)
            latent = latent_loss(pred, target, valid)
            rec = reconstruction_loss(self.decoder(pred), tactile['change'], tactile['change_mask'])
            nce = contrastive_loss(pred, target, valid, tactile['segment'], tactile['frame'], tactile['change'])
            return latent+lambda_rec*rec+lambda_nce*nce, {'latent': latent.detach(), 'reconstruction': rec.detach(), 'nce': nce.detach()}
        demo = self.baseline.prepare_action(batch)
        y = self.handoff(prefix)
        tau = torch.rand(len(y), device=y.device).clamp_min(1e-5)*0.4
        x, velocity = refinement_bridge(demo, y, tau)
        compact = self.resampler(pred)
        output = self.expert(x, tau, prefix, compact,
                             mask_prefix=self.stage == 'expert' and self.stage2_mask_prefix)
        loss = velocity_loss(output, velocity, batch['action_is_pad'])
        # Action supervision alone trains the resampler. Stage 2 freezes the
        # predictor; Stage 3 keeps the action-gradient path through it open.
        return loss, {'refinement': loss.detach()}

    @torch.no_grad()
    def continue_action_chunk(self, prefix, compact, handoff, *, mask_prefix=None):
        """Integrate a fixed handoff; shared by deployment and paired diagnostics."""
        if mask_prefix is None:
            mask_prefix = self.stage == 'expert' and self.stage2_mask_prefix
        x = handoff.clone()
        for step in range(4):
            tau = torch.full((len(x),), 0.4-step*0.1, device=x.device)
            x = x-0.1*self.expert(x, tau, prefix, compact, mask_prefix=mask_prefix)
        return x[..., :9]

    @torch.no_grad()
    def predict_action_chunk(self, batch, depth, *, condition='predicted', noise=None):
        self.eval()
        prefix = encode_prefix(self.baseline, batch)
        return self.predict_action_chunk_from_prefix(prefix, depth, condition=condition, noise=noise)

    @torch.no_grad()
    def predict_action_chunk_from_prefix(self, prefix, depth, *, condition='predicted', noise=None):
        """Predict a fresh 50-action horizon using explicitly managed context."""
        self.eval()
        if condition == 'current':
            pred = self.representation(depth)
        else:
            pred = self.predicted_tokens(depth, prefix)
            if condition == 'shuffle':
                if len(pred) < 2:
                    raise ValueError('Shuffling requires multiple observations')
                pred = pred.roll(1, 0)
            elif condition not in ('predicted', 'zero'):
                raise ValueError(condition)
        pred = self.resampler(pred)
        if condition == 'zero':
            # Zero the actual expert condition, including learned resampler biases.
            pred = torch.zeros_like(pred)
        return self.continue_action_chunk(prefix, pred, self.handoff(prefix, noise))

    def added_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('baseline.')}

    def load_added_state_dict(self, state, *, allow_legacy_grounding=False):
        expected = set(self.added_state_dict())
        obsolete_heads = ('resampler.reconstruction_attention.', 'resampler.reconstruction_head.')
        if any(k.startswith(obsolete_heads) for k in state):
            if not allow_legacy_grounding:
                raise ValueError('Removing compression heads requires a validated encoder/grounding checkpoint')
            state = {k: v for k, v in state.items() if not k.startswith(obsolete_heads)}
        if isinstance(self.resampler, SpatialTactileResampler) and not any(k.startswith('resampler.') for k in state):
            if not allow_legacy_grounding:
                raise ValueError('Adding a resampler requires a validated encoder/grounding checkpoint')
            # The resampler is unused in grounding, so its new initialization does
            # not affect the restored grounding optimizer or trained representation.
            state = {**state, **{k: v for k, v in self.added_state_dict().items() if k.startswith('resampler.')}}
        if (self.expert.implementation == 'shared_ffn_lora_v1'
                and int(state.get('expert.smolvla_initialization_version', -1)) == 1):
            if not allow_legacy_grounding:
                raise ValueError('Copied expert migration requires a validated encoder/grounding checkpoint; '
                                 'trained dense experts cannot be silently converted to LoRA')
            # The copied expert was frozen and unused in grounding. Keep the new
            # adapters at initialization; transfer ALL learned tactile components.
            transferred = {k: v for k, v in state.items() if not k.startswith('expert.')}
            if set(transferred) != {k for k in expected if not k.startswith('expert.')}:
                raise ValueError('Legacy grounding tactile checkpoint keys mismatch')
            state = {**transferred, **{k: v for k, v in self.added_state_dict().items()
                                      if k.startswith('expert.')}}
        if set(state) != expected:
            raise ValueError(f'Refinement checkpoint keys mismatch: {set(state)^expected}')
        self.load_state_dict(state, strict=False)
