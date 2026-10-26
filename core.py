"""Shared model, transforms, and reproducibility helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 37


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model


def make_transform(size: int = 224, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(round(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        return self.transform(image.convert('RGB')), int(label)


def device_for(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(pretrained=False)
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved


@torch.inference_mode()
def predict_image(model, image: Image.Image, size: int, device: torch.device, names: list[str]):
    x = make_transform(size)(image.convert('RGB')).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    vals, inds = probs.topk(3)
    return [(names[int(i)], float(v)) for v, i in zip(vals, inds)]
