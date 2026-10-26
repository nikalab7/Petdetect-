"""Choose input-resolution optimization on validation; test once, and analyze errors."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from core import TransformedSubset, device_for, load_checkpoint, make_transform, set_seed


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    gold, guesses = [], []
    model.eval()
    for images, labels in loader:
        y = model(images.to(device)).argmax(1).cpu().tolist()
        gold.extend(labels.tolist())
        guesses.extend(y)
    return {
        'accuracy': float(accuracy_score(gold, guesses)),
        'macro_f1': float(f1_score(gold, guesses, average='macro', zero_division=0)),
    }, gold, guesses


@torch.inference_mode()
def benchmark(model, base, indices, size, device, trials=50):
    model.eval()
    durations = []
    sample_ids = [int(i) for i in indices[:min(trials, len(indices))]]
    if not sample_ids:
        raise ValueError('No images available for benchmarking')
    transform = make_transform(size)
    for j, idx in enumerate(sample_ids):
        image, _ = base[idx]
        start = time.perf_counter()
        x = transform(image.convert('RGB')).unsqueeze(0).to(device)
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if j >= 5:  # ignore warmup runs
            durations.append((time.perf_counter() - start) * 1000)
    return {'median_latency_ms': round(statistics.median(durations), 2),
            'measured_images': len(durations), 'device': str(device)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='artifacts')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    device = device_for(args.device)
    out = Path(args.out_dir)
    model, ckpt = load_checkpoint(out / 'best.pt', device)
    set_seed(ckpt['seed'])
    base_trainval = OxfordIIITPet(root=args.data_dir, split='trainval', target_types='category', download=True)
    labels = np.asarray(base_trainval._labels)
    _, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                   stratify=labels, random_state=ckpt['seed'])
    optimization = {}
    for size in (224, 160):
        val_ds = TransformedSubset(base_trainval, val_idx, make_transform(size))
        val_metrics, _, _ = evaluate(model, val_ds, device, args.batch_size)
        latency = benchmark(model, base_trainval, val_idx, size, device)
        optimization[str(size)] = {**val_metrics, **latency}
        print(f'Validation {size}px: {optimization[str(size)]}', flush=True)
    # Choose resolution on validation, NEVER using test labels.
    is_faster = optimization['160']['median_latency_ms'] < optimization['224']['median_latency_ms']
    acceptable_quality = optimization['160']['macro_f1'] >= optimization['224']['macro_f1'] - 0.02
    selected_size = 160 if is_faster and acceptable_quality else 224
    base_test = OxfordIIITPet(root=args.data_dir, split='test', target_types='category', download=True)
    test_ds = TransformedSubset(base_test, range(len(base_test)), make_transform(selected_size))
    test_metrics, gold, guesses = evaluate(model, test_ds, device, args.batch_size)
    names = ckpt['class_names']
    if list(base_test.classes) != list(names):
        raise RuntimeError('Train/test class order mismatch')
    matrix = confusion_matrix(gold, guesses, labels=list(range(len(names))))
    np.save(out / 'confusion_matrix.npy', matrix)
    errors = [{'dataset_index': i, 'expected': names[y], 'predicted': names[p]}
              for i, (y, p) in enumerate(zip(gold, guesses)) if y != p][:20]
    report = {
        'optimization': {'validation_resolution_comparison': optimization,
                         'selected_image_size': selected_size,
                         'selection_rule': '160px only if faster and validation macro-F1 falls <=0.02'},
        'held_out_test': {**test_metrics, 'count': len(gold)},
        'errors_sample': errors,
        'checkpoint_size_mb': round((out / 'best.pt').stat().st_size / 1_000_000, 3),
        'device': str(device),
        'note': 'Optimization selected on validation only; test evaluated once at chosen resolution.',
    }
    (out / 'eval_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    ckpt['image_size'] = selected_size
    torch.save(ckpt, out / 'best.pt')
    print('Final test:', report['held_out_test'])
    print('Saved:', out / 'eval_results.json')


if __name__ == '__main__':
    main()
