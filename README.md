# The Pareidolia Paradox - Final Submission

## Methodology Summary

### Handling `sun_azimuth_angle`
In this challenge, craters and mounds are visually near-mirror-images of each other under shadow reversal. To ensure our model correctly learns the underlying 3D geometry rather than just the 2D lighting direction, we handled the `sun_azimuth_angle` through a two-pronged approach:

1. **Spatial Canonicalization**: We used PyTorch's `torchvision.transforms.functional.rotate` to rotate every image by `-sun_azimuth_angle`. Before rotation, we padded the image with a constant gray value (`fill=128`) to prevent edge-gradient artifacts. After rotation, we took a center crop. This mathematically aligned all images so that the simulated sun position was perfectly normalized to a single canonical direction, meaning shadows for craters always appeared on the same side regardless of original lighting.

2. **Hybrid Feature Extraction**: We passed these geometrically canonicalized images through a pre-trained ResNet18 backbone (acting as a robust frozen feature extractor) to obtain 512-dimensional dense embeddings. To maximize predictive power and recover any residual global lighting context, we combined these CNN features with the original `sun_azimuth_angle` (and its sine/cosine components) and trained a robust `HistGradientBoostingClassifier` on the concatenated feature space.

## Repository Structure
- `train.py`: Script to extract features using the CNN and train the Gradient Boosting model.
- `inference.py`: Script to load the trained model, process the test set, and output `submission.csv`.
- `requirements.txt`: Required dependencies.

## Model Weights
*(Insert Link to Google Drive / Hugging Face Here)*

