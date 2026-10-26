"""Generate a short numerical report from actual training and evaluation records."""
import json
from pathlib import Path

out = Path('artifacts')
train = json.loads((out / 'train_results.json').read_text(encoding='utf-8'))
eval_ = json.loads((out / 'eval_results.json').read_text(encoding='utf-8'))

best = {}
for row in train['experiment_history']:
    stage = row['stage']
    if stage not in best or row['macro_f1'] > best[stage]['macro_f1']:
        best[stage] = row

rows = ['| Training stage | Epoch | Validation accuracy | Validation macro-F1 |', '|---|---:|---:|---:|']
for stage in ('head_only', 'fine_tune'):
    row = best[stage]
    rows.append(f"| {stage} | {row['epoch']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
res = eval_['optimization']['validation_resolution_comparison']
opt = ['| Input resolution | Validation accuracy | Validation macro-F1 | Median latency (ms) |', '|---|---:|---:|---:|']
for size in ('224', '160'):
    opt.append(f"| {size}px | {res[size]['accuracy']:.4f} | {res[size]['macro_f1']:.4f} | {res[size]['median_latency_ms']:.2f} |")
errors = eval_['errors_sample'][:5]
errs = ['| Official test index | Expected breed | Predicted breed |', '|---:|---|---|']
errs += [f"| {x['dataset_index']} | {x['expected']} | {x['predicted']} |" for x in errors]

report = f'''# PetBreed AI — measured technical report

## Scope and data
Predict one of 37 cat/dog breeds from a single photograph, using Oxford-IIIT Pet. Official `trainval` split was divided into stratified 80% train and 20% validation with seed {train['seed']}; official test set remained held out. Training set used {train['train_count']} examples and validation used {train['validation_count']}.

## Architecture and method
Pretrained MobileNetV3-Small chosen for modest training cost and CPU inference. Images are normalized with ImageNet statistics. Training uses random crop and horizontal flip; evaluation uses deterministic resizing and center crop. Stage 1 freezes pretrained backbone and learns the 37-class linear head; Stage 2 unfreezes the last two feature blocks at a lower learning rate. Cross-entropy loss and AdamW optimizer. Select checkpoint by highest validation macro-F1, not by test accuracy.

## Training experiments

{chr(10).join(rows)}

Selected checkpoint: **{train['selected_stage']}**, with best validation macro-F1 **{train['best_validation_macro_f1']:.4f}**.

## Optimization experiment
The same checkpoint was evaluated at two input sizes (all timings measured on **{eval_['device']}**, preprocessing and inference included). The lower resolution is selected only when faster and within 0.02 absolute macro-F1 on validation.

{chr(10).join(opt)}

Chosen input size: **{eval_['optimization']['selected_image_size']}px**. This is an input-resolution speed–quality experiment, not weight pruning or quantization. Any quality loss and any failure to speed up should be reported honestly.

## Held-out test
Evaluated the selected model once on the official test split (n={eval_['held_out_test']['count']}). **Accuracy {eval_['held_out_test']['accuracy']:.4f}; macro-F1 {eval_['held_out_test']['macro_f1']:.4f}**. Checkpoint file size: **{eval_['checkpoint_size_mb']:.2f} MB**.

## Error sample
The table contains the first five saved test mistakes, ordered by test index. They all have the same expected class in this run and are not a representative sample of all errors. See `REPORT.md` for the visually reviewed image observations and tentative explanations. Do not infer image-specific causes from class labels alone. `confusion_matrix.npy` contains the full per-class counts.

## Practical use and limitations
`streamlit run app.py` opens a CPU-compatible photo-upload app with top-3 breed predictions. `python predict.py <photo>` provides CLI inference. The model always chooses among 37 known breeds; unrelated images, multiple pets, and unseen breeds may yield overconfident outputs. Softmax values are not calibrated. No object localisation or segmentation is claimed.

## Future improvements
Investigate open-set rejection, calibration, additional augmentation, quantitative robustness testing and retraining with more varied images. Reproducibility instructions, environment and dependency requirements are in `README.md`.

## AI tool usage
See `AI_USAGE.md` for the project's AI assistance disclosure.
'''
Path('FINAL_REPORT.md').write_text(report, encoding='utf-8')
print('Wrote FINAL_REPORT.md — see REPORT.md for the visual error review.')
