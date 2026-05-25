import os
import pathlib
from PIL import Image
from glob import glob
from joblib import Parallel, delayed

IMG_PATH = "png_data"
pathlib.Path(IMG_PATH).mkdir(parents = True, exist_ok = True)

img_files = glob(os.path.join("S B ages", "*.jpg"))

def process_round_1(img_file):
    img_name = f"{pathlib.Path(img_file).stem}.png".replace("mako-", "").replace("s.png", ".png")
    out_path = os.path.join(IMG_PATH, img_name)
    if os.path.isfile(out_path):
        return

    img = Image.open(img_file)
    rgb_im = img.convert('RGB')
    rgb_im.save(out_path)

Parallel(n_jobs = 4)(delayed(process_round_1)(img_file) for img_file in img_files)

img_files = glob(os.path.join("S B ages round 2", "*.jpg"))

def process_round_2(img_file):
    img_name = f"{pathlib.Path(img_file).stem}.png".replace("mako-", "").replace("-", "_").replace("_pseudoreflected_SB", "")
    out_path = os.path.join(IMG_PATH, img_name)
    if os.path.isfile(out_path):
        return

    img = Image.open(img_file)
    rgb_im = img.convert('RGB')
    rgb_im.save(out_path)

Parallel(n_jobs = 4)(delayed(process_round_2)(img_file) for img_file in img_files)
