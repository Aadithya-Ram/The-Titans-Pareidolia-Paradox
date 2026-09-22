# The Pareidolia Paradox - Final Submission

## ?? Trained Model & Accuracy

| Metric                  | Result                            |
| ----------------------- | --------------------------------- |
| **Final Trained Model** | **Hybrid HistGradientBoostingClassifier (with ResNet18)** |
| **Task**                | Lunar Terrain Classification      |
| **Validation Accuracy** | **78.29%** (Balanced Accuracy)    |
| **Model Weights**       | `hist_gb_model.pkl`               |

*Note: Evaluated using a 5-Fold Stratified K-Fold CV. A formal Test Accuracy is not available as test labels are hidden.*

---

## Methodology Summary

### Data Preparation
- **Padding & Normalization**: Images were padded with 64 pixels (constant gray, `fill=128`) to prevent edge artifacts. We used ImageNet standard normalization (`mean=[0.485, 0.456, 0.406]`).
- **Rotation Canonicalization**: To handle the physics clue (`sun_azimuth_angle`), images were rotated by `-sun_azimuth_angle` using bilinear interpolation to perfectly canonicalize the lighting direction, followed by a 256x256 center crop.
- **Data Splitting**: 5-Fold Stratified K-Fold was used for robust cross-validation.

### Model Architecture
- **Feature Extractor**: A pre-trained `ResNet18` backbone (transfer learning from ImageNet) was used as a frozen feature extractor, dropping the final fully connected layer to output 512-dimensional embeddings.
- **Classification Head**: The final classifier is a `HistGradientBoostingClassifier`.
- **Hybrid Input**: The CNN features were concatenated with the raw `sun_azimuth_angle` (along with its sine and cosine) to maximize predictive power for the 2 classes (Crater vs. Mound).

### Training
- **Classifier Parameters**: The gradient boosting tree was trained with `random_state=42` and `max_iter=200`.
- **Thresholding**: We swept the decision threshold and identified `0.48` as the optimal cutoff for maximum Balanced Accuracy.

### Evaluation
- **Metric**: Balanced Accuracy.
- **Selection**: The model (`hist_gb_model.pkl`) is the final trained classifier resulting from fitting on the canonicalized feature space.

---

## ?? Hugging Face Model

The trained model checkpoint is hosted on Hugging Face for easy access and independent evaluation.

**Hugging Face:** [https://huggingface.co/harsh-chikane/lunar-terrain-classifier](https://huggingface.co/harsh-chikane/lunar-terrain-classifier)

Relevant model file:
`hist_gb_model.pkl`

---

## ?? Project Resources

* **?? Hugging Face Model:** [https://huggingface.co/harsh-chikane/lunar-terrain-classifier](https://huggingface.co/harsh-chikane/lunar-terrain-classifier)
* **?? Dataset/Data File:** [https://drive.google.com/drive/folders/1nSdNXvR0ZMMqPoxn-iijvV5gBZLIXd5p?usp=sharing](https://drive.google.com/drive/folders/1nSdNXvR0ZMMqPoxn-iijvV5gBZLIXd5p?usp=sharing)

---

## Project Overview
This project classifies 256x256 grayscale lunar surface crops into two categories:
- **Class 0 (Depth)**: Craters, holes, and surface depressions.
- **Class 1 (Rise)**: Mounds, hills, rocks, and boulders.

## Project Structure
- `train.py`: Script to extract features using the CNN and train the Gradient Boosting model.
- `inference.py`: Script to load the trained model, process the test set, and output `submission.csv`.
- `requirements.txt`: Required dependencies.

## Installation / Setup
```bash
pip install -r requirements.txt
```

## Usage / Inference
To generate predictions on the test set:
```bash
python inference.py
```
This will read from `Test/test_metadata.csv` and generate the `submission.csv`.

## Training
To train the model from scratch:
```bash
python train.py
```
This script will read from `Train/train_metadata.csv`, extract canonicalized ResNet18 features, and train the `hist_gb_model.pkl`.

