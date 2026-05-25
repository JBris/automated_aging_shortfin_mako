import pandas as pd
import os
import numbers
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
CLEAN_DIR = os.path.join(IN_DIR,"png_data")
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

img_files = glob(os.path.join(CLEAN_DIR, "*.png"))

for img_file in img_files:
    img_name = Path(img_file).stem

    for sub_string in ["_a_", "_b_", "_c_", "_d_"]:
        if sub_string not in img_name:
            continue
        new_name = img_name.replace(sub_string, sub_string[1:])
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_file = os.path.join(CLEAN_DIR, f"{new_name}.png")
        cv2.imwrite(output_file, image)
        os.remove(img_file)
        print(f"Cleaning: {img_file}")
