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
from swag_experiments.swag.posteriors import SWAG
from swag_experiments.swag.models.preresnet import PreResNet, PreResNet20NoAug
from swag_experiments.swag.models.wide_resnet import WideResNet
from swag_experiments.swag.utils import bn_update

M = 5
DIM = 256
BATCH_SIZE = 16
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

class ResNetFineTuned(PreResNet):
  def __init__(
      self,
      num_classes=10, depth=110, planes=(16, 32, 64), input_size=32, skip_connections=True
  ):
    super().__init__(num_classes, depth, planes, input_size, skip_connections)
    # self.base_model = torchvision.models.get_model("resnet50", weights="DEFAULT")
    # for param in self.base_model.parameters():
    #   param.requires_grad = True

    # num_feats = self.base_model.fc.in_features
    # self.base_model.fc = nn.Sequential(
    #   nn.Linear(num_feats, 512),
    #   nn.SiLU(),
    #   nn.Linear(512, 512),
    #   nn.SiLU(),
    #   nn.Linear(512, 256),
    #   nn.SiLU(),
    #   nn.Linear(256, 256),
    #   nn.SiLU(),
    #   nn.Dropout(p=0.2),
    #   nn.Linear(256, 256),
    #   nn.SiLU(),
    #   nn.Linear(256, 256),
    #   nn.SiLU(),
    #   nn.Linear(256, 256),
    #   nn.SiLU(),
    #   nn.Linear(256, 256),
    #   nn.SiLU(),
    #   nn.Linear(256, 128),
    #   nn.Linear(128, 2)
    # )

    # self.base_model = PreResNet()
    # self.base_model.fc =  nn.Linear(10, 256)
    self.fc = nn.Sequential(
       nn.Linear(1024, 32),
       nn.Linear(32, 2)
    )
    self.jitter = 1e-6

  def forward(self, x):
      x = self.conv1(x)

      x = self.layer1(x)  # 32x32
      x = self.layer2(x)  # 16x16
      x = self.layer3(x)  # 8x8
      x = self.bn(x)
      x = self.relu(x)

      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      mean = x[:, 0]
      std = F.softplus(x[:, 1]) + self.jitter
      return torch.distributions.Normal(mean, std)

#######################################3
# To GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = SWAG(ResNetFineTuned,
  subspace_type="covariance", subspace_kwargs={'max_rank': 15})
model.to('cuda')

from torchinfo import summary as model_summary
# model_summary(model, input_size=(BATCH_SIZE ,3, DIM, DIM))

#######################################
transformations = torchvision.transforms.Compose([
  # torchvision.transforms.RandomHorizontalFlip(p = 0.5),
  # torchvision.transforms.RandomVerticalFlip(p = 0.5),
  # torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
  # torchvision.transforms.RandomInvert(),
  # torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
  torchvision.transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
  )
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 14, pin_memory = True)

##########################################################################

learning_rate = 1e-3
decay = 0
weight_decay = 0
lr_gamma = 0.25

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = lr_gamma, step_size = 20, verbose = True)

CHECKPOINT_DIR=os.path.join("data", "image_reg", "resnet50_nll_swag")
Path(CHECKPOINT_DIR).mkdir(parents = True, exist_ok = True)

#############################################################

def compute_loss(y_hat, y):
    neg_log_likelihood = -y_hat.log_prob(y)
    loss = torch.mean(neg_log_likelihood)
    return loss

for m in range(M):
   print(f"Model {m + 1}")

   model_dir = os.path.join(CHECKPOINT_DIR, f"model_{m}", "checkpoint")
   Path(model_dir).mkdir(parents = True, exist_ok = True)
        # self.register_buffer('mean', torch.zeros(self.num_parameters))
        # self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        # self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

  #  model.load_state_dict(torch.load(os.path.join(model_dir, "resnet50_nll.pth")) )
   model = torch.load(os.path.join(model_dir, "resnet50_swag"))
   print(model.base_model)
   print(model.mean)
   print(model)
   model.eval()

   test_mse = 0
   test_mae = 0

   test_labels = []
   test_preds = []
   print(model.sample())
   print(model.sample())
   model.sample()
   bn_update(train_loader, model)

   with torch.no_grad():
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      y_hats = model(inputs)
      y_preds = y_hats.mean.detach().cpu().numpy()

      test_labels.extend((labels.cpu().squeeze().numpy()))
      test_preds.extend((y_preds))

   y_test = test_labels
   plt.figure(figsize=(10,10))
   plt.scatter(y_test, test_preds, c='crimson')
   p1 = max(max(test_preds), max(y_test))
   p2 = min(min(test_preds), min(y_test))
   plt.title("Test results of non-Bayesian ResNet50 model.")
   plt.plot([p1, p2], [p1, p2], 'b-')
   plt.xlabel('Actual', fontsize=15)
   plt.ylabel('Predictions', fontsize=15)
   plt.axis('equal')
   pred_img_path = os.path.join(model_dir, "preds_uncertainty.png")
   plt.savefig(pred_img_path)
