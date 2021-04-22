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
imagenet_df = get_run_df(base_dir)
imagenet_df.sort_values('train_sz', inplace=True)
imagenet_df

# %%
_imagenet_df = imagenet_df.loc[imagenet_df['key'].isin(
    ['0.001-200', '0.01-100', '0.1-25', '1.0-20'])]

# %%
base_dir = 's3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a'
spacenet_sup_df = get_run_df(base_dir)
spacenet_sup_df.sort_values('train_sz', inplace=True)
spacenet_sup_df

# %%
base_dir = 's3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-unsupervised/4-20-2021a/'
spacenet_unsup_df = get_run_df(base_dir)
spacenet_unsup_df.sort_values('train_sz', inplace=True)
spacenet_unsup_df

# %%
fig, ax = plt.subplots(1, 1)
train_sz = [0.001, 0.01, 0.1, 1.0]
ax.plot(
    _imagenet_df['train_sz'].to_numpy(),
    _imagenet_df['building_f1'].to_numpy(),
    label='supervised imagenet')
ax.plot(
    spacenet_sup_df['train_sz'].to_numpy(),
    spacenet_sup_df['building_f1'].to_numpy(),
    label='supervised spacenet')
ax.set_xticks(train_sz)
ax.set_xlabel('proportion of dataset for training')
ax.set_ylabel('building f1')
ax.set_xscale('log')
ax.legend()

# %%
def plot_training_logs(log_uris, labels):
    for log_uri, label in zip(log_uris, labels):
        log_df = pd.read_csv(StringIO(file_to_str(log_uri)))
        epoch = log_df['epoch'].to_numpy()
        building_f1 = log_df['building_f1'].to_numpy()
        plt.plot(epoch, building_f1, label=label)
        plt.xlabel('epoch')
        plt.ylabel('building f1')
        plt.title('Trained on 1%')
        plt.legend()

spacenet_uri = 's3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a/0.01/100/train/log.csv'
imagenet_uri = 's3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.01/100/train/log.csv'
plot_training_logs([spacenet_uri, imagenet_uri], ['spacenet supervised', 'imagenet supervised'])

# %%
