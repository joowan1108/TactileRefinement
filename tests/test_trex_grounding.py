import torch
import copy
import pytest
from torch.nn import functional as F

from tactile_refinement.losses import n0_contrastive_loss, n0_grounding_loss, coarse_change_target
from tactile_refinement.representation import TactileRepresentation, Predictor, CoarseChangeHead


def test_grounding_budget_extension_preserves_every_other_setting():
    from tactile_refinement.train import validate_resume_config
    before = {'stage2_mask_prefix': True, 'batch_size': 16, 'stages': {
        'grounding': {'updates': 10000, 'learning_rate': 1e-4},
        'expert': {'updates': 20000, 'learning_rate': 5e-5}}}
    extended = copy.deepcopy(before)
    extended['stages']['grounding']['updates'] = 100000
    validate_resume_config(before, extended, 'grounding')
    for key, value in [('learning_rate', 1e-3), ('updates', 9000)]:
        invalid = copy.deepcopy(extended)
        invalid['stages']['grounding'][key] = value
        with pytest.raises(ValueError):
            validate_resume_config(before, invalid, 'grounding')
    for key, value in [('batch_size', 8), ('stage2_mask_prefix', False)]:
        with pytest.raises(ValueError):
            validate_resume_config(before, dict(extended, **{key: value}), 'grounding')
    with pytest.raises(ValueError):
        validate_resume_config(before, extended, 'expert')


def test_full_grid_keeps_spatial_slots_without_pooling(monkeypatch):
    torch.set_num_threads(2)
    representation = TactileRepresentation(full_grid=True).eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pooling was called inside the representation')
    monkeypatch.setattr(F, 'adaptive_avg_pool2d', forbidden)
    with torch.no_grad():
        tokens = representation(torch.randn(1, 2, 240, 320))
        assert tokens.shape == (1, 600, 256)
        predicted = Predictor(64, positions_per_finger=300)(tokens, torch.randn(1, 7, 64),
                                                          torch.ones(1, 7, dtype=torch.bool))
        assert predicted.shape == tokens.shape
        assert CoarseChangeHead()(predicted).shape == (1, 2, 60, 80)


def test_future_stage_budget_can_increase_when_initializing_from_grounding():
    from tactile_refinement.train import validate_resume_config
    saved = {'batch_size': 16, 'stages': {
        'grounding': {'updates': 100000, 'learning_rate': 1e-4},
        'expert': {'updates': 20000, 'learning_rate': 5e-5},
        'joint': {'updates': 10000, 'learning_rate': 1e-5}}}
    extended = copy.deepcopy(saved)
    extended['stages']['expert']['updates'] = 100000
    extended.update(expert_implementation='shared_ffn_lora_v1', lora_rank=32, lora_alpha=32,
                    resampler_tokens_per_finger=16)
    validate_resume_config(saved, extended, 'grounding')
    for stage in ('expert', 'joint'):
        with pytest.raises(ValueError):
            validate_resume_config(saved, extended, stage)
    bad_lr = copy.deepcopy(extended)
    bad_lr['stages']['expert']['learning_rate'] = 1e-3
    with pytest.raises(ValueError):
        validate_resume_config(saved, bad_lr, 'grounding')
    with pytest.raises(ValueError):
        validate_resume_config(saved, dict(extended, batch_size=8), 'grounding')


def test_n0_uses_mean_tokens_unit_cosine_and_symmetric_ce():
    torch.manual_seed(1)
    pred = torch.randn(4, 5, 8, requires_grad=True)
    target = torch.randn(4, 5, 8)
    valid = torch.tensor([True, False, True, True])
    loss, _ = n0_contrastive_loss(pred, target, valid)
    similarity = F.normalize(pred[valid].mean(1), dim=-1) @ F.normalize(target[valid].mean(1), dim=-1).T
    expected = (F.cross_entropy(similarity, torch.arange(3)) +
                F.cross_entropy(similarity.T, torch.arange(3)))/2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert pred.grad[1].count_nonzero() == 0
    assert pred.grad[valid].abs().sum() > 0


def test_coarse_target_signed_invalid_pixel_and_endpoint_masks():
    change = torch.tensor([[[[-1., 9.], [.5, 1.]]], [[[8., 8.], [8., 8.]]]])
    mask = torch.tensor([[[[True, False], [True, True]]], [[[True, True], [True, True]]]])
    valid = torch.tensor([True, False])
    target, coarse_mask = coarse_change_target(change, mask, valid, (1, 1))
    torch.testing.assert_close(target.flatten(), torch.tensor([1/6, 0.]))
    assert coarse_mask.flatten().tolist() == [True, False]
    pred = torch.randn(2, 3, 4, requires_grad=True)
    decoded = torch.zeros(2, 1, 1, 1, requires_grad=True)
    loss, parts = n0_grounding_loss(pred, pred.detach(), decoded, change, mask, valid)
    torch.testing.assert_close(loss, torch.tensor(1/6))  # one valid NCE anchor => CE=0
    assert set(parts) == {'nce', 'reconstruction', 'retrieval_top1', 'valid_targets', 'prediction_feature_std'}
    loss.backward()
    assert decoded.grad[1].count_nonzero() == 0
    empty, _ = n0_grounding_loss(pred, pred.detach(), decoded, change, mask, ~torch.ones(2, dtype=torch.bool))
    assert float(empty.detach()) == 0 and torch.isfinite(empty)
