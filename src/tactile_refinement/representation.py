"""T-Rex DeformEncoder topology and our ordered two-finger token bottleneck.

Architecture reference (released weights loaded by the full-grid training recipe):
https://github.com/ZhuoyangLiu2005/T-Rex/blob/f88e10c61da123c68bf0927cf4860bc97a0381f3/qwen_vla/DeformAE.py
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class DeformEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3, bias=False),
                                  resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        self.reshape_layer1 = self.projection(128)
        self.reshape_layer2 = self.projection(256)

    @staticmethod
    def projection(channels):
        return nn.Sequential(nn.Conv2d(channels, 128, 3, padding=1),
                             nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.reshape_layer1(self.layer2(self.layer1(self.stem(x))))
        return self.reshape_layer2(self.layer3(x))


class TactileRepresentation(nn.Module):
    def __init__(self, width=256, *, full_grid=False):
        super().__init__()
        self.encoder = DeformEncoder()
        self.projector = nn.Linear(128, width)
        self.full_grid = full_grid

    def forward(self, maps):
        b, fingers, h, w = maps.shape
        if fingers != 2:
            raise ValueError('Expected ordered left/right maps')
        features = self.encoder(maps.reshape(b*2, 1, h, w))
        if self.full_grid:
            if features.shape[-2:] != (15, 20):
                raise ValueError('Full-grid tactile inputs must produce a 15x20 feature grid')
            tokens = features.flatten(2).transpose(1, 2)
        else:
            tokens = F.adaptive_avg_pool2d(features, (4, 4)).flatten(2).transpose(1, 2)
        return self.projector(tokens).reshape(b, 2*tokens.shape[1], -1)


class CoarseChangeHead(nn.Module):
    """Predict a signed 60x80 change field per finger from 600 predicted tokens."""
    def __init__(self, width=256):
        super().__init__()
        self.project = nn.Linear(width, 64)
        self.spatial = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 32, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, tokens):
        b, length, _ = tokens.shape
        if length != 600:
            raise ValueError('Expected two ordered 15x20 tactile grids')
        x = self.project(tokens).reshape(b*2, 15, 20, 64).permute(0, 3, 1, 2)
        x = self.spatial(x)
        x = F.interpolate(x, size=(60, 80), mode='bilinear', align_corners=False)
        return self.output(x).reshape(b, 2, 60, 80)


class TokenDecoder(nn.Module):
    """Decode exclusively through the projected tokens; output is signed/unbounded."""
    def __init__(self, width=256):
        super().__init__()
        self.unproject = nn.Linear(width, 128)
        self.blocks = nn.Sequential(nn.Conv2d(128, 64, 5, padding=2), nn.ReLU(),
                                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                    nn.Conv2d(64, 32, 5, padding=2), nn.ReLU())
        self.output = nn.Conv2d(32, 1, 5, padding=2)

    def forward(self, tokens, size=(240, 320)):
        b = tokens.shape[0]
        x = self.unproject(tokens).reshape(b*2, 4, 4, 128).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(15, 20), mode='bilinear', align_corners=False)
        x = self.blocks(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.output(x).reshape(b, 2, *size)


class Predictor(nn.Module):
    def __init__(self, context_width, width=256, *, positions_per_finger=16):
        super().__init__()
        self.prefix_adapter = nn.Linear(context_width, width)
        self.sensor = nn.Parameter(torch.randn(2, 1, width)*0.02)
        self.position = nn.Parameter(torch.randn(1, positions_per_finger, width)*0.02)
        self.queries = nn.Parameter(torch.randn(1, 2*positions_per_finger, width)*0.02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(
            width, 8, width*4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        ) for _ in range(2)])
        self.norm = nn.LayerNorm(width)

    def forward(self, current, context, context_mask):
        if current.shape[1] != self.queries.shape[1]:
            raise ValueError('Tactile input length does not match predictor queries')
        ordered = current + (self.sensor+self.position).reshape(1, self.queries.shape[1], -1)
        memory = torch.cat((ordered, self.prefix_adapter(context)), 1)
        padding = torch.cat((torch.zeros(current.shape[:2], dtype=torch.bool, device=current.device),
                             ~context_mask.bool()), 1)
        x = self.queries.expand(current.shape[0], -1, -1)
        for block in self.blocks:
            x = block(x, memory, memory_key_padding_mask=padding)
        return self.norm(x)


class SpatialTactileResampler(nn.Module):
    """Compress predicted grids independently per finger using action supervision."""
    def __init__(self, width=256, tokens_per_finger=16):
        super().__init__()
        if not isinstance(tokens_per_finger, int) or not 1 <= tokens_per_finger < 300:
            raise ValueError('Resampler tokens_per_finger must be an integer in [1, 299]')
        self.tokens_per_finger = tokens_per_finger
        self.row = nn.Parameter(torch.randn(15, 1, width)*0.02)
        self.column = nn.Parameter(torch.randn(1, 20, width)*0.02)
        self.finger = nn.Parameter(torch.randn(2, 1, width)*0.02)
        self.queries = nn.Parameter(torch.randn(1, tokens_per_finger, width)*0.02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(
            width, 8, width*4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        ) for _ in range(2)])
        self.norm = nn.LayerNorm(width)

    def positions(self):
        return (self.row+self.column).reshape(1, 300, -1)+self.finger

    def forward(self, predicted):
        b, length, width = predicted.shape
        if length != 600:
            raise ValueError('Resampler expects two ordered 15x20 predicted grids (600 tokens)')
        memory = (predicted.reshape(b, 2, 300, width)+self.positions()).reshape(b*2, 300, width)
        # Treat fingers as independent batch items: no attention crosses fingers.
        x = (self.queries+self.finger).unsqueeze(0).expand(b, -1, -1, -1)
        x = x.reshape(b*2, self.tokens_per_finger, width)
        for block in self.blocks:
            x = block(x, memory)
        return self.norm(x).reshape(b, 2*self.tokens_per_finger, width)
