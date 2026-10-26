# PetBreed AI — End-to-End Computer Vision

37-class cat/dog breed classification with a pretrained **MobileNetV3-Small**. Uses **Oxford-IIIT Pet** and PyTorch. Training is run in Google Colab; [open the training notebook in Colab](https://colab.research.google.com/github/nikalab7/Petdetect-/blob/main/PetBreedAI_Training.ipynb).

## Requirements

- Python 3.11 or 3.12 recommended; CPU works, optional free CUDA GPU speeds training.
- Internet required for initial dataset and pretrained ImageNet weight downloads.
- First-time dataset download needs several hundred MB of disk space.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train in Google Colab

Open `PetBreedAI_Training.ipynb` in Google Colab, choose a GPU runtime when available, then run the cells in order. Sign in to Google and allow the notebook to save its checkpoint and results in a new timestamped folder under `MyDrive/PetBreedAI/runs/`. The notebook records its Git commit and Python package versions with each run. Colab's free GPU availability can vary; the notebook also supports CPU.

The initial run uses 3 head-only epochs and 3 fine-tuning epochs. Review its validation accuracy, macro-F1, and loss before deciding whether to train for more epochs. Each run receives a separate output folder.

## Assess and run the demo

```bash
# Run after copying the completed training output into artifacts/
python evaluate.py
python make_report.py
python -m pytest -q
streamlit run app.py
```

To assess a checkpoint downloaded from Drive, place `best.pt` and `train_results.json` in `artifacts/`. Then run `python evaluate.py`. The training notebook downloads Oxford-IIIT Pet and ImageNet weights, splits the official `trainval` set **stratified 80/20** with seed 42, and runs two trainable stages:

1. `head_only`: ImageNet backbone frozen; train new 37-class classifier.
2. `fine_tune`: unfreeze the final two feature blocks, train them at smaller learning rate.

After every epoch, training loss, validation loss, accuracy and macro-F1 are recorded and saved to the run folder. The best validation macro-F1 determines `best.pt`, which is updated during training so it is retained if a Colab session stops. The official `test` split is never used for tuning.

`evaluate.py` compares 224px vs 160px **inference input resolution** on validation accuracy, macro-F1 and measured latency. It adopts 160px only if latency improves and validation macro-F1 decreases by no more than 0.02. It then tests the selected setup on the held-out official test split. Evaluation saves a confusion matrix (`.npy`) and sample mistakes.

Keep `train_results.json`, `eval_results.json`, `confusion_matrix.npy` and `best.pt` together for reproducibility.

## Demo

`streamlit run app.py` launches a local image-upload UI. It displays the image and top 3 breed predictions. The CLI alternative is `python predict.py path/to/photo.jpg`. No paid API, hosted application, or GPU required for inference. For a reviewer, include model weights via GitHub Releases or a downloadable link, or precise model-download/reproduction instructions; **Git ignores weight files by default**.

## Project files

- `core.py`: model, transforms, checkpoint handling, device selection.
- `train.py`: dataset splitting, fine-tuning, model checkpoint, learning history.
- `evaluate.py`: validation-selected optimization, held-out test metrics, errors, latency.
- `app.py`, `predict.py`: working demo and CLI.
- `tests/test_core.py`: basic no-download smoke tests.
- `PetBreedAI_Training.ipynb`: minimal Google Colab training notebook.
- `REPORT.md`: technical report template; `make_report.py` creates `FINAL_REPORT.md` after real measurements.
- `AI_USAGE.md`: AI tooling disclosure.

## Limitations

- Classification assumes the image contains one recognizable cat/dog; it neither localizes nor segments objects.
- Softmax is uncalibrated; new breeds, multiple animals, or unrelated images can be misclassified.
- Small data, class-level ambiguity, varying photographs and CPU/GPU timings limit conclusions.
- Validation and test quality differ; never represent validation metrics as held-out test results.
