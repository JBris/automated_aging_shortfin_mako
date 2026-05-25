from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pandas as pd
from pathlib import Path
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Manager

M = 5
DIM = 256
BATCH_SIZE = 16
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = "data"
FILE_DIR = os.path.join(DATA_DIR, "image_ages.csv")

IN_DIR = "in"
AUGMENT_DIR = os.path.join(IN_DIR, "sb_augment")
TARGET_DIR = AUGMENT_DIR

CHECKPOINT_DIR = os.path.join("data", "image_reg", "resnet50_deep_ensemble_nll")
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

img_files = glob(os.path.join(TARGET_DIR, "*.png"))
df = pd.read_csv(FILE_DIR)

df["count"].replace(to_replace=0, value=1, inplace=True)
df["count"] = df["count"].astype("float32")

count_dict = {df["name"][i]: df["count"][i] for i in df.index}

X_train, y_train = [], []

for img_file in img_files:
    X_train.append(img_file)
    img_name = Path(img_file).stem
    img_name = img_name.rpartition("_")[0]
    y_train.append(count_dict[img_name])

X_train = np.array(X_train)
y_train = np.array(y_train)

manager = Manager()
shared_cache = manager.dict()

class VertebraeDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, shared_cache=None):
        self.X = X
        self.y = torch.from_numpy(y.astype(float)).float().flatten()
        self.shared_cache = shared_cache

    def __getitem__(self, index):
        image_path = self.X[index]
        label = self.y[index]

        image = None
        if self.shared_cache is not None:
            image = self.shared_cache.get(image_path)

        if image is None:
            image = torchvision.io.read_file(image_path)
            image = torchvision.io.decode_png(image, mode=torchvision.io.ImageReadMode.RGB).float()
            image = torchvision.transforms.Resize((DIM, DIM))(image)
            image = image / 255.0

            if self.shared_cache is not None:
                self.shared_cache[image_path] = image

        return image, label

    def __len__(self):
        return len(self.X)

train_dataset = VertebraeDataset(X_train, y_train, shared_cache)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=14,
    pin_memory=True
)

class ResNetFineTuned(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = torchvision.models.get_model(
            "resnet50",
            weights="DEFAULT"
        )

        num_feats = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.SiLU(),

            nn.Linear(128, 2)
        )

        self.jitter = 1e-6

    def forward(self, x):
        x = self.base_model(x)

        mean = x[:, 0]
        std = F.softplus(x[:, 1]) + self.jitter

        return torch.distributions.Normal(mean, std)

def compute_loss(y_dist, y):
    return -y_dist.log_prob(y).mean()

models = []

for m in range(M):
    print(f"\n===== Training model {m+1}/{M} =====")

    model = ResNetFineTuned().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.25
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for inputs, y in train_loader:
            inputs, y = inputs.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            dist = model(inputs)
            loss = compute_loss(dist, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        scheduler.step()

        print(f"Model {m+1} | Epoch {epoch+1} | "
              f"Loss: {total_loss / len(train_loader.dataset):.4f}")

    model_path = os.path.join(CHECKPOINT_DIR, f"model_{m}.pth")
    torch.save(model.state_dict(), model_path)

    models.append(model)

def ensemble_predict(models, loader, device):
    all_means = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)

        means = []

        with torch.no_grad():
            for model in models:
                model.eval()
                dist = model(inputs)
                means.append(dist.mean.cpu().numpy())

        means = np.stack(means, axis=0)

        all_means.append(means)
        all_labels.append(labels.numpy())

    all_means = np.concatenate(all_means, axis=1)
    y_true = np.concatenate(all_labels)

    mean_pred = np.mean(all_means, axis=0)
    epistemic_unc = np.std(all_means, axis=0)

    return y_true, mean_pred, epistemic_unc

y_test, y_pred, y_unc = ensemble_predict(models, train_loader, DEVICE)

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))

plt.plot([p1, p2], [p1, p2], 'b-')

plt.title("Deep Ensemble ResNet50 (NLL)")
plt.xlabel("Actual")
plt.ylabel("Predicted Mean")
plt.axis("equal")

pred_img_path = os.path.join(CHECKPOINT_DIR, "preds.png")
plt.savefig(pred_img_path)
plt.close()
