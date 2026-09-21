import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
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
            
    features = np.array(features)
    angles_list = np.array(angles_list)
    X = np.hstack([features, angles_list])
    return X

def main():
    print("Loading test data...")
    test_df = pd.read_csv("Test/test_metadata.csv")
    
    print("Extracting features...")
    X_test = extract_features(test_df, "Test/eval_images")
    
    print("Loading model weights...")
    with open("hist_gb_model.pkl", "rb") as f:
        clf = pickle.load(f)
        
    print("Running inference...")
    probs = clf.predict_proba(X_test)[:, 1]
    
    # We found 0.48 was the optimal threshold during CV tuning
    test_df["label"] = (probs > 0.48).astype(int)
    
    print("Generating submission.csv...")
    submission = test_df[["image_id", "label"]]
    submission.to_csv("submission.csv", index=False)
    print("Done! Predictions saved to submission.csv")

if __name__ == "__main__":
    main()

