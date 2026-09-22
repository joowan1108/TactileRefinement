import torch
from torch import nn

from tactile_refinement.attention_audit import cross_attention_probabilities, ResamplerAttentionRecorder
from tactile_refinement.representation import SpatialTactileResampler


def test_shadow_readout_matches_pytorch_attention():
    torch.manual_seed(1)
    attention = nn.MultiheadAttention(32, 8, dropout=0, batch_first=True).eval()
    q, k = torch.randn(2, 16, 32), torch.randn(2, 300, 32)
    with torch.inference_mode():
        _, expected = attention(q, k, k, need_weights=True, average_attn_weights=False)
        result = cross_attention_probabilities(attention, q, k)
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-6)


def test_recording_does_not_change_resampler_outputs():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    resampler = SpatialTactileResampler(width=32).eval()
    dense = torch.randn(1, 600, 32)
    with torch.inference_mode():
        original = resampler(dense)
        recorder = ResamplerAttentionRecorder(resampler)
        captured = resampler(dense)
        torch.testing.assert_close(original, captured, atol=0, rtol=0)
        weights = recorder.probabilities()
        assert weights.shape == (2, 2, 8, 16, 300)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 8, 16))
        recorder.close()
        resampler(dense)
        assert not recorder.handles
