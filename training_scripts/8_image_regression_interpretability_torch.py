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
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from captum.attr import IntegratedGradients
from captum.attr import GradientShap, DeepLift
from captum.attr import Occlusion, FeatureAblation
from captum.attr import NoiseTunnel, Saliency
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

M = 5
DIM = 256
BATCH_SIZE = 32
INITIAL_SHAPE = (DIM, DIM, 3)
EPOCHS = 1000
SEED = 100
FILE_EXT = ".png"

DEVICE = "cuda"
DATA_DIR = "data"
FILE_DIR = os.path.join(DATA_DIR,"image_ages.csv")
IN_DIR = "in"
CLEAN_DIR = os.path.join(IN_DIR,"sb")
AUGMENT_DIR = os.path.join(IN_DIR,"sb_augment")

use_clean_dir = True
if use_clean_dir:
  TARGET_DIR = CLEAN_DIR
else:
  TARGET_DIR = AUGMENT_DIR

img_files = glob(os.path.join(TARGET_DIR, "*.png"))
df = pd.read_csv(FILE_DIR)
df["count"].replace(to_replace = 0, value = 1, inplace = True)
df["count"] = df["count"].astype("float32")

n_images = len(img_files)
print(f"Total number of images: {n_images}")

count_dict = {}
for i in df.index:
  count_dict[df["name"][i]] = df["count"][i]

X_train = []
y_train = []
for img_file in img_files:
  X_train.append(img_file)
  img_name = Path(img_file).stem

  if not use_clean_dir:
    img_name = img_name.rpartition("_")[0]

  y_train.append(count_dict[img_name])

X_train = np.array(X_train)
y_train = np.array(y_train)


image = torchvision.io.read_file(X_train[0])
image = torchvision.io.decode_png(image, mode =
  torchvision.io.ImageReadMode.RGB).float()
image = torchvision.transforms.Resize((DIM, DIM))(image)

#######################################
## Model


# model = torchvision.models.resnet50(pretrained = True)
model = torchvision.models.get_model("resnet50", weights="DEFAULT")
# Freeze weights
for param in model.parameters():
    param.requires_grad = True

# Replace final layer
num_feats = model.fc.in_features

model.fc = nn.Sequential(
  nn.Linear(num_feats, 512),
  nn.SiLU(),
  nn.Linear(512, 512),
  nn.SiLU(),
  nn.Linear(512, 256),
  nn.SiLU(),
  nn.Linear(256, 256),
  nn.SiLU(),
  nn.Linear(256, 256),
  nn.SiLU(),
  nn.Linear(256, 256),
  nn.SiLU(),
  nn.Linear(256, 256),
  nn.SiLU(),
  nn.Linear(256, 256),
  nn.SiLU(),
  nn.Linear(256, 128),
  nn.Linear(128, 1)
)

#######################################3
# To GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = model.to(device)

#######################################
transformations = torchvision.transforms.Compose([
  # torchvision.transforms.RandomHorizontalFlip(p = 0.5),
  # torchvision.transforms.RandomVerticalFlip(p = 0.5),
  # torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
  # torchvision.transforms.RandomInvert(),
  # torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
  # torchvision.transforms.Normalize(
  #   mean = [0.485, 0.456, 0.406],
  #   std = [0.229, 0.224, 0.225]
  # )
])

from multiprocessing import Manager

manager = Manager()
shared_cache = manager.dict()

class VertebraeDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transformations = None, shared_cache = None):
        self.X = X
        self.y = torch.from_numpy(y.astype(float)).float().flatten()
        self.shared_cache = shared_cache
        self.transformations = transformations

    def __getitem__(self, index):
        image_path = self.X[index]
        label = self.y[index]
        image = None

        if shared_cache is not None:
          image = shared_cache.get(image_path)

        if image is None:
          image = torchvision.io.read_file(image_path)
          image = torchvision.io.decode_png(image, mode =
            torchvision.io.ImageReadMode.RGB).float()
          image = torchvision.transforms.Resize((DIM, DIM))(image)
          image = image / 255

          if shared_cache is not None:
            shared_cache[image_path] = image

        if self.transformations is not None:
            image = self.transformations(image)

        return (image, label)

    def __len__(self):
        return len(self.X)

train_dataset = VertebraeDataset(X = X_train, y = y_train, transformations = transformations, shared_cache = shared_cache)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 14, pin_memory = True)

##########################################################################

learning_rate = 1e-3
decay = 0
weight_decay = 0
lr_gamma = 0.25

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = lr_gamma, step_size = 20, verbose = True)

CHECKPOINT_DIR=os.path.join("data", "image_reg", "resnet50_mse")
Path(CHECKPOINT_DIR).mkdir(parents = True, exist_ok = True)

#############################################################

for m in range(M):
   print(f"Model {m + 1}")
   model_dir = os.path.join(CHECKPOINT_DIR, f"model_{m}", "checkpoint")
   Path(model_dir).mkdir(parents = True, exist_ok = True)

   model.load_state_dict(torch.load(os.path.join(model_dir, "resnet50_mse.pth")))
   model.eval()
   integrated_gradients = IntegratedGradients(model)
   ig = IntegratedGradients(model)
   ig_nt = NoiseTunnel(ig)
   dl = DeepLift(model)
   gs = GradientShap(model)
   fa = FeatureAblation(model)

   i = 300
   for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    for img_in in inputs:
      i += 1
      img_in = img_in.unsqueeze(0)
      attributions_ig = integrated_gradients.attribute(img_in, n_steps=1)
      default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                        [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

      _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    np.transpose(img_in.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    show_colorbar=True,
                                    sign='positive',
                                    outlier_perc=1)

      Path(os.path.join(model_dir, "ig")).mkdir(parents = True, exist_ok = True)
      plt.savefig(os.path.join(model_dir, "ig", f"ig_feat_{i}.png"))

      attributions_ig_nt = ig_nt.attribute(img_in, nt_samples=1, nt_type='smoothgrad_sq')
      _ = viz.visualize_image_attr(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(img_in.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            method='heat_map',
                                            cmap=default_cmap,
                                            show_colorbar=True)

      Path(os.path.join(model_dir, "nt")).mkdir(parents = True, exist_ok = True)
      plt.savefig( os.path.join(model_dir, "nt", f"nt_feat_{i}.png"))

      torch.manual_seed(0)
      np.random.seed(0)


      # Defining baseline distribution of images
      rand_img_dist = torch.cat([img_in * 0, img_in * 1])

      attributions_gs = gs.attribute(img_in,
                                                n_samples=2,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist)
      _ = viz.visualize_image_attr(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(img_in.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            method='heat_map',
                                            cmap=default_cmap,
                                            show_colorbar=True,
                                            sign='positive',
                                            outlier_perc=1)

      Path(os.path.join(model_dir, "shap")).mkdir(parents = True, exist_ok = True)
      plt.savefig(os.path.join(model_dir, "shap", f"shap_feat_{i}.png"))

      saliency = Saliency(model)
      grads = saliency.attribute(img_in)
      grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
      _ = viz.visualize_image_attr(grads, original_image = np.transpose(img_in.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    show_colorbar=True,
                                    sign='positive',
                                    outlier_perc=1)

      Path(os.path.join(model_dir, "sal")).mkdir(parents = True, exist_ok = True)
      plt.savefig(os.path.join(model_dir, "sal", f"sal_feat_{i}.png"))
