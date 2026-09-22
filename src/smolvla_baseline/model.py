import json
from pathlib import Path

import torch
import draccus
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .data import IMAGE_KEYS


LANGUAGE_LORA_TARGETS = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def _enable_requested_training(model, recipe):
    """Apply the requested language strategy and audit its trainable surface."""
    if recipe['language_training'] == 'full_dense':
        if recipe.get('language_lora', False):
            raise ValueError('Full language training cannot enable LoRA')
        if any('lora_' in name for name, _ in model.named_parameters()):
            raise RuntimeError('LoRA parameters found in a dense fine-tuning run')
        return model
    if recipe['language_training'] != 'lora_only' or not recipe.get('language_lora', False):
        raise ValueError(f"Unsupported language training strategy: {recipe['language_training']}")
    lora = LoraConfig(
        r=recipe['language_lora_rank'],
        lora_alpha=recipe['language_lora_alpha'],
        lora_dropout=recipe['language_lora_dropout'],
        bias='none',
        target_modules=LANGUAGE_LORA_TARGETS,
    )
    model = get_peft_model(model, lora)

    dense_prefixes = (
        'base_model.model.model.vlm_with_expert.vlm.model.vision_model.',
        'base_model.model.model.vlm_with_expert.vlm.model.connector.',
        'base_model.model.model.vlm_with_expert.lm_expert.',
        'base_model.model.model.state_proj.',
        'base_model.model.model.action_in_proj.',
        'base_model.model.model.action_out_proj.',
        'base_model.model.model.action_time_mlp_in.',
        'base_model.model.model.action_time_mlp_out.',
    )
    for name, parameter in model.named_parameters():
        if name.startswith(dense_prefixes):
            parameter.requires_grad_(True)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not any('lora_' in name and '.text_model.' in name for name in trainable):
        raise RuntimeError('No language-backbone LoRA parameters were enabled')
    if any('lora_' in name and '.text_model.' not in name for name in trainable):
        raise RuntimeError('LoRA was applied outside the language backbone')
    for fragment in ('vision_model.', 'lm_expert.', 'state_proj.', 'action_in_proj.', 'action_out_proj.'):
        if not any(fragment in name and 'lora_' not in name for name in trainable):
            raise RuntimeError(f'Required dense training module is frozen: {fragment}')
    frozen_language_base = [
        name for name, parameter in model.named_parameters()
        if '.text_model.' in name and 'lora_' not in name and parameter.requires_grad
    ]
    if frozen_language_base:
        raise RuntimeError(f'Base language weights unexpectedly trainable: {frozen_language_base[:3]}')
    return model


def build_model(recipe, manifest, checkpoint=None):
    model_dir = snapshot_download(recipe['pretrained_model'], revision=recipe['model_revision'], allow_patterns=['config.json', 'model.safetensors'])
    raw_config = json.loads((Path(model_dir) / 'config.json').read_text())
    assert raw_config.pop('type') == 'smolvla'
    config = draccus.decode(SmolVLAConfig, raw_config)
    config.vlm_model_name = snapshot_download(recipe['vlm_model'], revision=recipe['vlm_revision'], allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
    config.device = recipe['device']
    config.input_features = {'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(9,))}
    config.input_features.update({key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for key in IMAGE_KEYS})
    config.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    config.chunk_size = recipe['chunk_size']
    config.n_action_steps = recipe['n_action_steps']
    config.resize_imgs_with_padding = (recipe['image_size'], recipe['image_size'])
    config.freeze_vision_encoder = recipe['freeze_vision_encoder']
    config.train_expert_only = recipe['train_expert_only']
    config.train_state_proj = recipe['train_state_proj']
    # The complete SmolVLA checkpoint supplies the backbone weights as well.
    config.load_vlm_weights = False
    config.push_to_hub = False
    model = SmolVLAPolicy.from_pretrained(model_dir, config=config, strict=True)
    model = _enable_requested_training(model, recipe)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
    stats = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in stat.items()} for key, stat in manifest['stats'].items()}
    pre, post = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    return model, pre, post


def load_checkpoint(path):
    # All our checkpoints contain only tensors and primitive containers.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['recipe']['tactile_inputs'] or checkpoint['recipe']['tactile_auxiliary_loss']:
        raise ValueError('Checkpoint is not an action-only-loss, no-tactile baseline')
    return checkpoint
