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
FILE_DIR = os.path.join(DATA_DIR,"sb_ages.csv")
IN_DIR = "in"
CLEAN_DIR = os.path.join(IN_DIR,"sb")
BLACK_N_WHITE_DIR = os.path.join(IN_DIR,"sb_bw")

df = pd.read_csv(FILE_DIR)
img_files = glob(os.path.join(CLEAN_DIR, "*.png"))
Path(BLACK_N_WHITE_DIR).mkdir(parents=True, exist_ok=True)

for img_file in img_files:
    img_name = Path(img_file).name

    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128,128))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    output_file = os.path.join(BLACK_N_WHITE_DIR, img_name)
    cv2.imwrite(output_file, image)
