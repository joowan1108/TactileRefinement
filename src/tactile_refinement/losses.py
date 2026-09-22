"""Masked representation/prediction and correctly scaled continuation losses."""
import torch
from torch.nn import functional as F


def reconstruction_loss(pred, target, mask, contact_weight=4.0):
    weights = mask.float() * (1 + contact_weight * (target.abs() > 1e-3))
    return ((pred.float()-target.float()).abs()*weights).sum() / weights.sum().clamp_min(1)


def latent_loss(pred, target, valid):
    error = F.smooth_l1_loss(pred.float(), target.float(), reduction='none').mean((1, 2))
    return (error*valid).sum() / valid.sum().clamp_min(1)


def n0_contrastive_loss(pred, target, valid):
    """N0 Eq. 3-4: token mean, L2 normalization, symmetric batch CE, no temperature.

    Invalid horizon endpoints are removed. All remaining examples are in-batch
    negatives, as in the published objective (no legacy negative filtering).
    The target is detached by the caller. Accumulation does not pool negatives.
    """
    p = F.normalize(pred[valid].float().mean(1), dim=-1)
    t = F.normalize(target[valid].float().mean(1), dim=-1)
    if len(p) == 0:
        return pred.sum()*0, pred.detach().sum()*0
    # Keep near-identical cosine logits in float32 even under bf16 training.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        logits = p@t.T
    labels = torch.arange(len(p), device=pred.device)
    loss = (F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2
    accuracy = ((logits.argmax(1) == labels).float().mean()+
                (logits.argmax(0) == labels).float().mean())/2
    return loss, accuracy.detach()


def coarse_change_target(change, mask, valid, size=(60, 80)):
    """Valid-pixel area mean; full grid is retained inside the encoder itself."""
    mask = mask.bool() & valid[:, None, None, None]
    fraction = F.adaptive_avg_pool2d(mask.float(), size)
    total = F.adaptive_avg_pool2d(change.float()*mask, size)
    target = total/fraction.clamp_min(1e-8)
    return target, fraction > 0


def n0_grounding_loss(pred, target, decoded, change, mask, valid, lambda_rec=1.0):
    nce, accuracy = n0_contrastive_loss(pred, target, valid)
    coarse, coarse_mask = coarse_change_target(change, mask, valid, decoded.shape[-2:])
    rec = ((decoded.float()-coarse).abs()*coarse_mask).sum()/coarse_mask.sum().clamp_min(1)
    with torch.no_grad():
        pooled = pred[valid].float().mean(1)
        feature_std = pooled.std(0, correction=0).mean() if len(pooled) else pred.detach().sum()*0
    return nce+lambda_rec*rec, {'nce': nce.detach(), 'reconstruction': rec.detach(),
                               'retrieval_top1': accuracy, 'valid_targets': valid.sum().detach(),
                               'prediction_feature_std': feature_std}


def contrastive_loss(pred, target, valid, segment, frame, change, horizon=50, temperature=0.1):
    """Symmetric in-batch InfoNCE; exclude overlapping/near-zero/duplicate negatives."""
    b = len(pred)
    active = valid.bool() & (change.float().abs().flatten(1).amax(1) > 1e-3)
    allowed = active[:, None] & active[None, :]
    overlap = (segment[:, None] == segment[None, :]) & ((frame[:, None]-frame[None, :]).abs() <= horizon)
    flat = change.float().flatten(1)
    # Actual map difference, not merely magnitude, identifies indistinguishable targets.
    distance = torch.cdist(flat, flat, p=2) / flat.shape[1]**0.5
    allowed &= ~overlap & (distance > 1e-4)
    eye = torch.eye(b, dtype=torch.bool, device=pred.device)
    anchors = allowed.any(1) & active
    allowed |= eye
    p, t = F.normalize(pred.float().flatten(1), dim=-1), F.normalize(target.float().flatten(1), dim=-1)
    logits = (p @ t.T / temperature).masked_fill(~allowed, -1e4)
    labels = torch.arange(b, device=pred.device)
    error = (F.cross_entropy(logits, labels, reduction='none') +
             F.cross_entropy(logits.T, labels, reduction='none'))*0.5
    return (error*anchors).sum()/anchors.sum().clamp_min(1)


def refinement_bridge(demo, handoff, tau, tau_s=0.4):
    alpha = tau[:, None, None]/tau_s
    return (1-alpha)*demo + alpha*handoff.detach(), (handoff.detach()-demo)/tau_s


def velocity_loss(pred, target, action_is_pad, action_dim=9):
    error = (pred[..., :action_dim].float()-target[..., :action_dim].float()).square().mean(-1)
    valid = ~action_is_pad.bool()
    # Equal weighting per observation, including shorter terminal chunks.
    per_sample = (error*valid).sum(1)/valid.sum(1).clamp_min(1)
    return per_sample.mean()
