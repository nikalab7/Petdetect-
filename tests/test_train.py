import json

import numpy as np
import torch
from PIL import Image
from torch import nn

import train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()),
        )
        self.classifier = nn.Linear(4, 37)

    def forward(self, images):
        features = self.features(images).mean(dim=(2, 3))
        return self.classifier(features)


def test_training_keeps_frozen_batch_norm_stable_and_updates_fine_tuned_block():
    model = TinyModel()
    batch = [(torch.randn(4, 3, 12, 12), torch.tensor([0, 1, 2, 3]))]

    train.configure_training(model, 'head_only')
    early_bn = model.features[1]
    original_mean = early_bn.running_mean.clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    train.train_epoch(model, batch, optimizer, torch.device('cpu'))
    torch.testing.assert_close(early_bn.running_mean, original_mean)

    train.configure_training(model, 'fine_tune')
    original_mean = early_bn.running_mean.clone()
    tuned_bn = model.features[-1][1]
    tuned_mean = tuned_bn.running_mean.clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    train.train_epoch(model, batch, optimizer, torch.device('cpu'))
    torch.testing.assert_close(early_bn.running_mean, original_mean)
    assert not torch.equal(tuned_bn.running_mean, tuned_mean)


def test_fine_tuning_starts_from_best_head_and_saves_validation_history(monkeypatch, tmp_path):
    class TinyPet:
        def __init__(self, **kwargs):
            self._labels = np.repeat(np.arange(37), 5)
            self.classes = [f'breed_{i}' for i in range(37)]

        def __len__(self):
            return len(self._labels)

        def __getitem__(self, index):
            return Image.new('RGB', (12, 12), 'white'), int(self._labels[index])

    model = TinyModel()
    starts = []
    score_call = 0

    monkeypatch.setattr(train, 'OxfordIIITPet', TinyPet)
    monkeypatch.setattr(train, 'make_model', lambda pretrained: model)
    monkeypatch.setattr(train, 'make_transform', lambda size=224, train=False: lambda image: torch.zeros(3, 12, 12))

    def fake_train_epoch(current, loader, optimizer, device):
        fine_tuning = any(p.requires_grad for p in current.features.parameters())
        starts.append(current.classifier.weight.detach().clone())
        with torch.no_grad():
            current.classifier.weight.fill_(2.0 if fine_tuning else 1.0)
        return 0.5

    def fake_score(current, loader, device):
        nonlocal score_call
        score_call += 1
        return {
            'loss': 0.8,
            'accuracy': 0.5,
            'macro_f1': 0.7 if score_call == 1 else 0.6,
        }

    monkeypatch.setattr(train, 'train_epoch', fake_train_epoch)
    monkeypatch.setattr(train, 'score', fake_score)
    monkeypatch.setattr('sys.argv', [
        'train.py', '--out-dir', str(tmp_path), '--head-epochs', '1', '--tune-epochs', '1',
        '--batch-size', '16', '--device', 'cpu',
    ])

    train.main()

    assert torch.count_nonzero(starts[0]) > 0
    torch.testing.assert_close(starts[1], torch.ones_like(starts[1]))
    checkpoint = torch.load(tmp_path / 'best.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(checkpoint['state_dict']['classifier.weight'], torch.ones_like(starts[1]))
    assert checkpoint['best_stage'] == 'head_only'
    history = json.loads((tmp_path / 'train_results.json').read_text(encoding='utf-8'))
    assert [row['stage'] for row in history['experiment_history']] == ['head_only', 'fine_tune']
    assert all('validation_loss' in row for row in history['experiment_history'])
