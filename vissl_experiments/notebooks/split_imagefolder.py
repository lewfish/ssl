# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile

import glob
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from rastervision.pipeline.file_system import (
    make_dir, list_paths, file_to_json, file_to_str, download_if_needed)

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", None)

# %%
data_dir = '/opt/data/research/ssl/resisc45/'
class_names = [basename(p) for p in glob.glob(join(data_dir, '**'))]
output_dir = '/opt/data/research/ssl/resisc45-split/'
image_paths = glob.glob(join(data_dir, '**/*.jpg'))
train_prop = 0.6
val_prop = 0.2
train_subsets = [0.1, 0.2]
nb_images = len(image_paths)
nb_train = int(round(train_prop * nb_images))
nb_val = int(round(val_prop * nb_images))

# %%
random.seed(1234)
random.shuffle(image_paths)
train_paths = image_paths[0:nb_train]
val_paths = image_paths[nb_train:nb_train+nb_val]
test_paths = image_paths[nb_train+nb_val:]

def copy_split(split, image_paths, output_dir):
    split_dir = join(output_dir, split)
    make_dir(split_dir)
    for image_path in image_paths:
        dst_path = join(split_dir, *image_path.split('/')[-2:])
        make_dir(dst_path, use_dirname=True)
        shutil.copyfile(image_path, dst_path)

for class_name in class_names:

for train_subset in train_subsets:
    copy_split('train', train_paths, output_dir)
    copy_split('val', val_paths, output_dir)
    copy_split('test', test_paths, output_dir)
