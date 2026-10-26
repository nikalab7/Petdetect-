# PetBreed AI - measured technical report

## Problem and scope

The project classifies a single cat or dog image into one of 37 Oxford-IIIT Pet breeds. It does not locate or segment an animal, and it has no rejection class for unfamiliar breeds or non-pet images.

## Data and preprocessing

Training used the official Oxford-IIIT Pet `trainval` split, divided into a stratified 80% training / 20% validation split with seed 42: 2,944 training images and 736 validation images. The official held-out test split contains 3,669 images and was evaluated after model and input-size selection.

Training augmentation uses random resized crops and horizontal flips. Validation and test use deterministic resize and center crop, followed by ImageNet normalization.

## Model and training results

The model is ImageNet-pretrained MobileNetV3-Small with a 37-output classifier. The run used three head-only epochs followed by three fine-tuning epochs. Fine-tuning unfreezes the final two feature blocks and uses a lower learning rate. The checkpoint was selected by validation macro-F1.

| Training stage | Best epoch | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: | ---: |
| Head-only | 3 | 0.8587 | 0.8590 |
| Fine-tuning | 3 | 0.8886 | 0.8894 |

Fine-tuning improved validation accuracy by 0.0299 and macro-F1 by about 0.0305 over the best head-only epoch. The completed Drive run is `/content/drive/MyDrive/PetBreedAI/runs/20261026_155151_utc`.

## Input-resolution comparison

The same trained checkpoint was measured on the Colab CPU. Latency is the median per-image preprocessing plus inference time; it is not a measurement of the user's local CPU.

| Input size | Validation accuracy | Validation macro-F1 | Median CPU latency |
| --- | ---: | ---: | ---: |
| 224px | 0.8886 | 0.88945 | 11.04 ms |
| 160px | Pending retrieval from the saved `eval_results.json` | 0.82142 | 8.87 ms |

At 160px, latency improved by 2.17 ms (about 19.7%), while macro-F1 fell by 0.06803. The project rule allows a maximum 0.02 absolute macro-F1 decrease, so 160px does not qualify. The selected production resolution is **224px**. The visualization cell in `PetBreedAI_Training.ipynb` prints both recorded validation accuracies directly from the saved evaluation JSON; the 160px value will be copied here after that file is downloaded.

## Final held-out test

The selected 224px configuration was evaluated once on the official held-out test split:

- Accuracy: **0.86018**
- Macro-F1: **0.85872**
- Images: **3,669**

These test metrics are separate from validation metrics and were not used to choose the input resolution.

## Error analysis

The saved evaluation JSON contains test indices and expected/predicted breed names for measured mistakes. The new Colab inspection cell displays five corresponding original test images and saves a labeled montage. Image-specific causes will be documented after visually reviewing those five images; breed similarity, pose, occlusion, and background are hypotheses until confirmed by the images.

## Demo, limitations, and reproduction

The Streamlit app loads `artifacts/best.pt` on CPU and returns the top three known breeds. A softmax score is not a calibrated confidence. The model can still assign a known breed to unrelated images, multiple animals, or breeds outside its 37 classes. It performs classification only, without localization or segmentation.

The run's checkpoint, training history, evaluation JSON, confusion matrix, and example images are kept together in the Drive run folder. The notebook's inspection and download cells only read those completed outputs; they do not retrain or reevaluate the model. For local inference, place the final checkpoint at `artifacts/best.pt`, install `requirements.txt`, and run `streamlit run app.py` or `python predict.py path/to/photo.jpg`.

## AI tool use

See `AI_USAGE.md` for the project-specific disclosure of AI assistance and human-run experiments.
