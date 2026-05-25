import os
import pathlib
from PIL import Image
from glob import glob
from joblib import Parallel, delayed
import albumentations as A
import albumentations as A
import cv2
from glob import glob
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import imutils

IN_DIR = "in"
FILE_DIR = os.path.join(IN_DIR,"sb_individuals")
CLEAN_DIR = os.path.join(IN_DIR,"sb")

Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

img_files = glob(os.path.join(FILE_DIR, "*.png"))

for img_file in img_files:
    img_name = Path(img_file).stem.replace("-", "_").replace(" ", "").lower()
    if img_name.endswith("_copy"):
        img_name = img_name.replace("_copy", "_b")
    else:
        img_name += "_a"
    output_file = os.path.join(CLEAN_DIR, f"{img_name}.png")

    if os.path.isfile(output_file):
        continue

    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imutils.resize(image, width = 1000, inter = cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT)

    blurred = cv2.blur(image, (5,5))
    canny = cv2.Canny(blurred, 180, 180)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(output_file, cropped)
    print(f"Cleaning: {img_file}")
