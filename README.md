# PetBreed AI - End-to-End Computer Vision

A 37-class cat/dog breed classifier using a pretrained **MobileNetV3-Small**, PyTorch, and the Oxford-IIIT Pet dataset. The completed 224px checkpoint and measured run results are included under `artifacts/`. Local inference needs no dataset download and runs on CPU.

## Final measured run

The best checkpoint came from fine-tuning: validation Accuracy 0.88859 and Macro-F1 0.88945. The official held-out test reached Accuracy 0.86018 and Macro-F1 0.85872 on 3,669 images. The 160px input was 19.7% faster on the Colab CPU but lost 0.06803 Macro-F1, beyond the allowed 0.02 decrease, so the project selected 224px. See `REPORT.md` for the full comparison, error review, limitations, and confusion matrix.

## Install and launch locally

Clone the repository and enter its root:

```powershell
git clone https://github.com/nikalab7/Petdetect-.git
cd Petdetect-
```

Create and activate a Python 3.11 virtual environment in PowerShell, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Command Prompt, activate with `.venv\Scripts\activate.bat`. On macOS or Linux, use `python3.11 -m venv .venv` and `source .venv/bin/activate` instead.

The trained checkpoint is already at `artifacts/best.pt`; do not download the dataset or run training/evaluation to launch the demo.

The committed run outputs also include `artifacts/train_results.json`, `artifacts/eval_results.json`, `artifacts/confusion_matrix.npy`, `artifacts/confusion_matrix.png`, and `artifacts/run_metadata.json`. The five source test photos and their montage remain local and are not committed as dataset images.

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a JPG or PNG, and view the top three predictions. The CLI alternative is:

```powershell
python predict.py path/to/photo.jpg
```

A prediction is always one of the 37 known classes. Softmax scores are not calibrated confidence estimates.

## Verify the project

```powershell
python -m pytest -q
```

The tests run without downloading Oxford-IIIT Pet. The Colab notebook includes separate cells for new training runs and for viewing/downloading an already completed run. Do not run its training or evaluation cells to use the committed checkpoint.

## Training reproducibility

`PetBreedAI_Training.ipynb` records a new run's source commit, runtime, package versions, checkpoint, and per-epoch metrics in Google Drive. The completed run used seed 42 and three head-only plus three fine-tuning epochs. Training used the official `trainval` split with a stratified 80/20 train/validation split. The official test split was held out until input-size selection. The saved metadata identifies the completed run and its original environment.

## Main files

- `core.py`: MobileNetV3-Small, image transforms, checkpoint loading, and top-three prediction.
- `train.py`: train/validation split, two transfer-learning stages, metrics, and best-checkpoint saving.
- `evaluate.py`: validation-only input-size selection, one held-out test evaluation, confusion matrix, and saved error indices.
- `app.py`, `predict.py`: Streamlit upload demo and command-line inference.
- `PetBreedAI_Training.ipynb`: Colab training, completed-run inspection, and download cells.
- `REPORT.md`: measured results and cautious visual error analysis.
- `AI_USAGE.md`: accurate disclosure of AI assistance and human-run experiments.

## Limitations

The classifier does not locate or segment pets and does not reject unknown classes. Unseen breeds, multiple animals, unrelated images, and changes in lighting/background can produce incorrect predictions. Colab CPU latency may differ from latency on a local machine.
