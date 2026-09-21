import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import torchvision.transforms.functional as F
from tqdm import tqdm

def extract_features(df, img_dir):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    features = []
    angles_list = []
    labels = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {img_dir}"):
        img_path = os.path.join(img_dir, row["image_id"])
        img = Image.open(img_path).convert("L")
        
        angle = row["sun_azimuth_angle"]
        img = F.pad(img, padding=64, padding_mode="constant", fill=128)
        img = F.rotate(img, -angle, interpolation=transforms.InterpolationMode.BILINEAR)
        img = F.center_crop(img, 256)
        img = img.convert("RGB")
        
        img_t = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_t = img_t.cuda()
            
        with torch.no_grad():
            feat = model(img_t).squeeze().cpu().numpy()
            
        angles_list.append([angle, np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        features.append(feat)
        
        if "label" in row:
            labels.append(row["label"])
            
    features = np.array(features)
    angles_list = np.array(angles_list)
    X = np.hstack([features, angles_list])
    
    if labels:
        return X, np.array(labels)
    return X

def main():
    print("Loading training data...")
    train_df = pd.read_csv("Train/train_metadata.csv")
    
    print("Extracting features...")
    X_train, y_train = extract_features(train_df, "Train/train_images")
    
    print("Training model...")
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    clf.fit(X_train, y_train)
    
    print("Saving model weights...")
    with open("hist_gb_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Done! Model saved to hist_gb_model.pkl")

if __name__ == "__main__":
    main()

