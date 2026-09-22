"""Observe resampler cross-attention without changing its inference path."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def cross_attention_probabilities(module, query, key):
    """Shadow QK readout; original MHA still uses its unmodified fast path.

    Scores/softmax use float32 for plotting. Under mixed precision this can
    differ slightly from fused attention numerics; no readout enters the policy.
    """
    if (not module.batch_first or not module._qkv_same_embed_dim
            or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn):
        raise ValueError('Unexpected resampler attention configuration')
    width = module.embed_dim
    q_bias = None if module.in_proj_bias is None else module.in_proj_bias[:width]
    k_bias = None if module.in_proj_bias is None else module.in_proj_bias[width:2*width]
    q = F.linear(query, module.in_proj_weight[:width], q_bias)
    k = F.linear(key, module.in_proj_weight[width:2*width], k_bias)
    head_dim = width // module.num_heads
    q = q.reshape(len(q), q.shape[1], module.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(len(k), k.shape[1], module.num_heads, head_dim).transpose(1, 2)
    with torch.autocast(device_type=q.device.type, enabled=False):
        return ((q.float() @ k.float().transpose(-1, -2)) * head_dim**-.5).softmax(-1)


class ResamplerAttentionRecorder:
    """Single-observation recorder: layers x fingers x heads x queries x grid."""
    def __init__(self, resampler):
        self.predicted = None
        self.layers = {}
        self.handles = [resampler.register_forward_pre_hook(self._start)]
        for index, block in enumerate(resampler.blocks):
            def capture(module, args, kwargs, index=index):
                if kwargs.get('attn_mask') is not None or kwargs.get('key_padding_mask') is not None:
                    raise ValueError('Unexpected masked resampler cross-attention')
                query, key = args[:2]
                if len(query) != 2 or query.shape[1] != 16 or key.shape[1] != 300:
                    raise ValueError('Expected one observation with 16 queries per finger and 15x20 grids')
                self.layers[index] = cross_attention_probabilities(module, query, key).cpu()
            self.handles.append(block.multihead_attn.register_forward_pre_hook(capture, with_kwargs=True))

    def _start(self, module, args):
        self.predicted = args[0].detach()
        self.layers = {}

    def probabilities(self):
        if len(self.layers) != 2:
            raise RuntimeError('Both resampler layers must have been observed')
        return torch.stack([self.layers[i] for i in range(2)])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
