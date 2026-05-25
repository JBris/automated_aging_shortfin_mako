import albumentations as A
import cv2
from glob import glob
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import pandas as pd
import cv2
from glob import glob
from pathlib import Path
import imutils

DATA_DIR = "data"
FILE_DIR = os.path.join(DATA_DIR,"image_ages.csv")
IN_DIR = "in"
CLEAN_DIR = os.path.join(IN_DIR,"sb")
AUGMENT_DIR = os.path.join(IN_DIR,"sb_augment")

Path(AUGMENT_DIR).mkdir(parents=True, exist_ok=True)
N_REPLICATES = 25

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

img_files = glob(os.path.join(CLEAN_DIR, "*.png"))
df = pd.read_csv(FILE_DIR)

transform = A.Compose([
    A.ShiftScaleRotate(shift_limit = 0.01, scale_limit = (-0.025, 0), rotate_limit = 360, border_mode = cv2.BORDER_CONSTANT, value = 0, p = 1),
    A.Sharpen(p = 1, alpha = 1),
    A.CLAHE(p = 1),
    A.ChannelShuffle(p = 1),
    A.Flip(),
    A.Transpose(),
    A.RandomBrightnessContrast(p = 1, brightness_limit = 0.05, contrast_limit = 0.05),
    A.Perspective(p = 1, pad_mode = cv2.BORDER_CONSTANT),
    A.HueSaturationValue(hue_shift_limit = .05, sat_shift_limit = .05, val_shift_limit = .05, p = 1),
    A.OpticalDistortion(p = 1),
    A.RandomToneCurve(p = 1),
    A.MedianBlur(p = 0.25),
    A.RandomToneCurve(scale=0.025, p = 1),
    A.Sharpen(p = 0.5),
    # A.RGBShift(p = 1),
    A.CoarseDropout(p = 1),
    A.GaussianBlur(p = .5),
    A.GaussNoise(p = .5),
])

i = 0
print("Augmenting images...")
for img_file in img_files:
    print(f"Augmenting: {img_file}")
    img_name = Path(img_file).stem
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imutils.resize(image, width = 1000, inter = cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT)

    for replicate in range(1, N_REPLICATES + 1):
        output_file = os.path.join(AUGMENT_DIR, f"{img_name}_{replicate}.png")
        if os.path.isfile(output_file):
            continue

        augmented_image = transform(image=image)['image']
        blurred = cv2.blur(augmented_image, (5,5))
        canny = cv2.Canny(blurred, 100, 100)
        pts = np.argwhere(canny>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)
        cropped = augmented_image[y1:y2, x1:x2]
        cv2.imwrite(output_file, cropped)
