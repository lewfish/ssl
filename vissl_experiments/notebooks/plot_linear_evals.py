# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile
import json
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
def get_acc(metrics_uri):
    metric_str = file_to_str(metrics_uri).split('\n')[-2]
    metrics_dict = json.loads(metric_str)
    acc = metrics_dict['test_accuracy_list_meter']['top_1']['res5avg']
    return acc

pretraining = [
    'imagenet/resisc moco (quad/50ep)',
    'imagenet moco',
    'imagenet supervised',
    'imagenet/resisc moco (single/20ep)',
]
eval_uris = [
    's3://research-lf-dev/ssl/vissl/test-output/05-26-2021e',
    's3://research-lf-dev/ssl/vissl/test-output/05-26-2021f',
    's3://research-lf-dev/ssl/vissl/test-output/05-26-2021g',
    's3://research-lf-dev/ssl/vissl/test-output/05-26-2021h',
]

accs = [get_acc(join(eval_uri, 'metrics.json')) for eval_uri in eval_uris]
df = pd.DataFrame({'pretraining': pretraining, 'accuracy': accs}).round(1)
df

# %%
ax = df.sort_values('accuracy').plot.bar(x='pretraining', y='accuracy', rot=0)
ax.set_ylim(87, 93)
ax.set_xlabel('pretraining strategy')
ax.set_title('resisc 45: linear evaluation')
ax.set_ylabel('accuracy')
ax.legend().remove()
ax.figure.savefig('/opt/data/research/ssl/vissl/ssl-advantage.png')
