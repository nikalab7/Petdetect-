"""Train and compare head-only and partial fine-tuning on Oxford-IIIT Pet."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from core import TransformedSubset, device_for, make_model, make_transform, set_seed


def train_epoch(model, loader, optimizer, device):
    model.train()
    for module in model.features.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


@torch.inference_mode()
def score(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    gold, predicted = [], []
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        total_count += len(labels)
        predicted.extend(logits.argmax(1).cpu().tolist())
        gold.extend(labels.cpu().tolist())
    return {
        'loss': total_loss / max(total_count, 1),
        'accuracy': float(accuracy_score(gold, predicted)),
        'macro_f1': float(f1_score(gold, predicted, average='macro', zero_division=0)),
    }


def configure_training(model, stage: str):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if stage == 'fine_tune':
        for p in model.features[-2:].parameters():
            p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='artifacts')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--tune-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)  # safe default on Windows
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--max-train', type=int, default=0, help='Optional stratified subset; 0=all')
    args = parser.parse_args()
    if args.head_epochs < 1 or args.tune_epochs < 1:
        parser.error('Each experiment requires at least one epoch')
    set_seed(args.seed)
    device = device_for(args.device)
    print(f'Device: {device}', flush=True)
    base = OxfordIIITPet(root=args.data_dir, split='trainval', target_types='category', download=True)
    labels = np.asarray(base._labels)
    indices = np.arange(len(base))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=args.seed)
    if args.max_train:
        if args.max_train < 74:
            parser.error('--max-train must be >= 74 to cover 37 breeds')
        if args.max_train < len(train_idx):
            train_idx, _ = train_test_split(train_idx, train_size=args.max_train,
                                          stratify=labels[train_idx], random_state=args.seed)
    train_ds = TransformedSubset(base, train_idx, make_transform(train=True))
    val_ds = TransformedSubset(base, val_idx, make_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    model = make_model(pretrained=True).to(device)
    stages = [('head_only', args.head_epochs, 1e-3), ('fine_tune', args.tune_epochs, 1e-4)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = []
    best_score = -1.0
    best_state = None
    best_stage = None
    best_head_state = None
    for stage, epochs, lr in stages:
        if stage == 'fine_tune' and best_head_state is not None:
            model.load_state_dict(best_head_state)
        configure_training(model, stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        stage_best_score = -1.0
        stage_best_state = None
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device)
            metrics = score(model, val_loader, device)
            record = {'stage': stage, 'epoch': epoch, 'train_loss': round(loss, 5),
                      'validation_loss': round(metrics['loss'], 5),
                      'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']}
            logs.append(record)
            print(json.dumps(record), flush=True)
            if stage == 'head_only' and metrics['macro_f1'] > stage_best_score:
                stage_best_score = metrics['macro_f1']
                stage_best_state = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})
            if metrics['macro_f1'] > best_score:
                best_score = metrics['macro_f1']
                best_state = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})
                best_stage = stage
            checkpoint = {
                'state_dict': best_state,
                'class_names': base.classes,
                'image_size': 224,
                'best_stage': best_stage,
                'seed': args.seed,
                'train_count': len(train_idx),
                'val_count': len(val_idx),
            }
            torch.save(checkpoint, out / 'best.pt')
            report = {
                'dataset': 'Oxford-IIIT Pet trainval', 'classes': len(base.classes),
                'device': str(device), 'seed': args.seed,
                'train_count': len(train_idx), 'validation_count': len(val_idx),
                'batch_size': args.batch_size,
                'max_epochs': {'head_only': args.head_epochs, 'fine_tune': args.tune_epochs},
                'torch_version': torch.__version__,
                'torchvision_version': torchvision.__version__,
                'experiment_history': logs, 'selected_stage': best_stage,
                'best_validation_macro_f1': best_score,
            }
            (out / 'train_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if stage == 'head_only' and stage_best_state is not None:
            best_head_state = stage_best_state
    print(f'Best checkpoint: {out / "best.pt"} ({best_stage})')
    print('Next: python evaluate.py')


if __name__ == '__main__':
    main()
