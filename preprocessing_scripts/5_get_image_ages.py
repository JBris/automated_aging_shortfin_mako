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

DATA_DIR = "data"
FILE_DIR = os.path.join(DATA_DIR,"sb_ages.csv")
IN_DIR = "in"
CLEAN_DIR = os.path.join(IN_DIR,"sb")

df = pd.read_csv(FILE_DIR)
img_files = glob(os.path.join(CLEAN_DIR, "*.png"))

img_count_list = []
for img_file in img_files:
    print(img_file)
    img_name = Path(img_file).stem.partition("_")[0]
    file_name = Path(img_file).stem
    count = df.loc[df['section'] == img_name]["count"].values.item()
    img_count_list.append(
        {"subject":img_name, "name": file_name, "file": img_file, "count": count}
    )

new_df = pd.DataFrame.from_dict(img_count_list)
out_file = os.path.join(DATA_DIR,"image_ages.csv")
new_df.to_csv(out_file, index=False)
