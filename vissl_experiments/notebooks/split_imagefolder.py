# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile

from tqdm import tqdm
import glob
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rastervision.pipeline.file_system import (
    make_dir, list_paths, file_to_json, file_to_str, download_if_needed, zipdir)

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", None)

# %%
data_dir = '/opt/data/research/ssl/resisc45/'
class_names = [basename(p) for p in glob.glob(join(data_dir, '**'))]
output_dir = '/opt/data/research/ssl/resisc45-split/'
train_prop = 0.6
val_prop = 0.2
train_subsets = [0.05, 1.0]

# %%
random.seed(1234)

class2train = {}
class2val = {}
class2test = {}
for class_name in class_names:
    image_paths = glob.glob(join(data_dir, class_name, '*.jpg'))
    random.shuffle(image_paths)

    nb_images = len(image_paths)
    nb_train = int(round(train_prop * nb_images))
    nb_val = int(round(val_prop * nb_images))

    class2train[class_name] = image_paths[0:nb_train]
    class2val[class_name] = image_paths[nb_train:nb_train+nb_val]
    class2test[class_name] = image_paths[nb_train+nb_val:]

def copy_split(split, image_paths, output_dir):
    split_dir = join(output_dir, split)
    make_dir(split_dir)
    for image_path in tqdm(image_paths, desc=split):
        dst_path = join(split_dir, *image_path.split('/')[-2:])
        make_dir(dst_path, use_dirname=True)
        shutil.copyfile(image_path, dst_path)

def concat_vals(adict):
    return [_v for k, v in adict.items() for _v in v]

# %%
for train_subset in train_subsets:
    _output_dir = join(output_dir, str(train_subset))
    print(f'Saving dataset to {_output_dir}...')

    _class2train = {}
    for class_name, image_paths in class2train.items():
        nb_images = len(image_paths)
        nb_train = int(round(train_subset * nb_images))
        _class2train[class_name] = image_paths[0:nb_train]

    copy_split('train', concat_vals(_class2train), _output_dir)
    copy_split('val', concat_vals(class2val), _output_dir)
    copy_split('test', concat_vals(class2test), _output_dir)
    zipdir(_output_dir, _output_dir + '.zip')

# %%
