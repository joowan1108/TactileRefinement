import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_encoder_pooling import spatial_input, ProbeDecoder, weighted_loss


def test_pooling_preserves_constants_but_changes_localized_information():
    constant=torch.ones(2,128,15,20)
    torch.testing.assert_close(spatial_input(constant,True),constant)
    local=torch.zeros_like(constant);local[:,:,7,10]=1
    assert spatial_input(local,False) is local
    pooled=spatial_input(local,True)
    assert pooled.shape==local.shape
    assert pooled.max()<local.max()
    assert torch.count_nonzero(pooled)>torch.count_nonzero(local)


def test_matched_decoders_have_equal_parameters_and_finite_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(42);a=ProbeDecoder()
    torch.manual_seed(42);b=ProbeDecoder()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
    features=torch.randn(1,128,15,20)
    pred=a(features)
    assert pred.shape==(1,1,240,320)
    truth=torch.zeros_like(pred);truth[:,:,100:102,200:202]=-.5
    valid=torch.ones_like(truth,dtype=torch.bool)
    loss=weighted_loss(pred,truth,valid);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
    assert features.grad is None
