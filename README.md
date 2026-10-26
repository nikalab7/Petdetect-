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

## Completed results and local demo

Training and the final held-out evaluation are complete for the Colab run recorded in `REPORT.md`. The selected model uses 224px input: 160px was about 19.7% faster on the Colab CPU, but its validation macro-F1 was 0.06803 lower and failed the 0.02 quality-loss limit. The held-out test reached 0.86018 accuracy and 0.85872 macro-F1 on 3,669 images.

The Colab notebook includes cells to visualize the saved confusion matrix and five actual test mistakes, then download the checkpoint and result files as one ZIP. These cells read the completed run and do not retrain or reevaluate it.

For local inference, place the final `best.pt` at `artifacts/best.pt` and run:

```bash
streamlit run app.py
# or
python predict.py path/to/photo.jpg
```

Do not run `train.py` or `evaluate.py` to launch the completed demo. Training and the one-time held-out evaluation are already finished. For an intentional future experiment, use a new run folder and a fresh evaluation record.

## Demo

`streamlit run app.py` launches a local image-upload UI. It displays the image and top 3 breed predictions. The CLI alternative is `python predict.py path/to/photo.jpg`. No paid API, hosted application, or GPU required for inference. The final checkpoint can be committed at `artifacts/best.pt`; dataset files and other model checkpoints stay out of Git.

## Project files

- `core.py`: model, transforms, checkpoint handling, device selection.
- `train.py`: dataset splitting, fine-tuning, model checkpoint, learning history.
- `evaluate.py`: validation-selected optimization, held-out test metrics, errors, latency.
- `app.py`, `predict.py`: working demo and CLI.
- `tests/test_core.py`: basic no-download smoke tests.
- `PetBreedAI_Training.ipynb`: minimal Google Colab training notebook.
- `REPORT.md`: current report with the measured run results; `make_report.py` can regenerate a numerical report from the saved JSON files in `artifacts/`.
- `AI_USAGE.md`: AI tooling disclosure.

## Limitations

- Classification assumes the image contains one recognizable cat/dog; it neither localizes nor segments objects.
- Softmax is uncalibrated; new breeds, multiple animals, or unrelated images can be misclassified.
- Small data, class-level ambiguity, varying photographs and CPU/GPU timings limit conclusions.
- Validation and test quality differ; never represent validation metrics as held-out test results.
