# Technical report — PetBreed AI

## Problem and scope
Classify **37 cat and dog breeds** from a single photo. Classification is an allowed computer-vision task in the assignment. No bounding boxes or segmentation masks are claimed.

## Dataset and preprocessing
Oxford-IIIT Pet. Official `trainval` split divided into 80% training / 20% validation, stratified, seed 42. Official `test` split held out. Train augmentation: random resized crop, horizontal flip, ImageNet mean/std. Validation/test: deterministic resize + center crop.

## Model and experiments
Pretrained MobileNetV3-Small chosen for manageable compute and model size. Stage 1 trains only the new 37-class head; stage 2 fine-tunes the last two backbone feature blocks at smaller learning rate. Compare validation macro-F1 and accuracy at every epoch. Select the best validation checkpoint.

**After running `python train.py`**, copy the actual values from `artifacts/train_results.json` into the table:

| Experiment | Validation accuracy | Validation macro-F1 | Epoch | Observation |
| --- | --- | --- | --- | --- |
| Head-only | NOT RUN | NOT RUN | NOT RUN | |
| Last blocks fine-tuned | NOT RUN | NOT RUN | NOT RUN | |

## Optimization
Compare the same trained checkpoint at **224px** and **160px** on validation accuracy, macro-F1, and median pre-processing + single-image inference latency. Only use 160px if faster with <=0.02 macro-F1 absolute decrease. Benchmark on identical hardware with warmup.

| Input size | Validation accuracy | Macro-F1 | Median latency (ms) |
| --- | --- | --- | --- |
| 224px | NOT RUN | NOT RUN | NOT RUN |
| 160px | NOT RUN | NOT RUN | NOT RUN |

Record the selected input resolution and explain the tradeoff. If the smaller resolution loses too much accuracy or is not faster, selecting 224px is an honest negative optimization result.

## Final held-out test
Selected configuration evaluated on the official test split only after experiments and optimization are decided. Accuracy: **NOT RUN**. Macro-F1: **NOT RUN**. Test count: **NOT RUN**. Model size (MB): **NOT RUN**.

## Error analysis
Inspect `artifacts/eval_results.json` (`errors_sample`) and `artifacts/confusion_matrix.npy`. Show 3 real misclassified examples with source dataset index, expected/predicted class, and plausible failure cause. Do not fabricate examples. Discuss breed similarity, occlusions/background, pose, and small sample size if observed.

## Production considerations and next steps
No paid external APIs. Inference runs locally in Streamlit on CPU. Model predicts one of the 37 known classes even for unrelated images; improve open-set rejection, confidence calibration and robustness if deploying publicly. More training epochs or model comparisons are possible future improvements.

## Reproducibility
See `README.md` for exact commands. Fix dataset split seed, dependency versions and log experiment settings. GPU/CPU, dataset size, timings and checkpoint release location must be documented after the real run.
