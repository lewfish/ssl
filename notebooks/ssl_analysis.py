# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from rastervision.pipeline.file_system import (
    make_dir, list_paths, file_to_json, file_to_str, download_if_needed)

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
def get_run_dirs(base_uri):
    run_dirs = list_paths(base_uri, ext='train/test_metrics.json')
    return [dirname(dirname(d)) for d in run_dirs]

# %%
def get_pipeline_options(key, pipeline_cfg_uri):
    """Returns a dict with the options/hyperparameters for a pipeline run."""
    pipeline_dict = file_to_json(pipeline_cfg_uri)
    solver = pipeline_dict['backend']['solver']
    data = pipeline_dict['backend']['data']

    num_epochs = solver['num_epochs']
    train_sz = data['train_sz_rel']

    opts = {
        'key': key,
        'num_epochs': num_epochs,
        'train_sz': train_sz,
    }
    return opts

# %%
def _get_run_df(run_dirs):
    # Combine options/hyperparams and metrics for a run into a df.
    dfs = []
    for run_dir in run_dirs:
        key = '-'.join(run_dir.split('/')[-2:])
        pipeline_cfg_uri = join(run_dir, 'pipeline-config.json')
        options = get_pipeline_options(key, pipeline_cfg_uri)
        metrics_uri = join(run_dir, 'train/test_metrics.json')
        metrics_dict = file_to_json(metrics_uri)
        df = pd.DataFrame()
        for ind, (key, val) in enumerate(options.items()):
            df.insert(ind, key, [val])
        df.insert(ind+1, 'building_f1', metrics_dict['building_f1'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_run_df(base_dir):
    run_dirs = get_run_dirs(base_dir)
    return _get_run_df(run_dirs)

# %%
base_dir = 's3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/'
df = get_run_df(base_dir)
