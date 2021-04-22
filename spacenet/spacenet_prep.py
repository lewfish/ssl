# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile
import re
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rastervision.pipeline.file_system import (
    make_dir, list_paths, file_to_json, file_to_str, download_if_needed, file_exists)

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
aois = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']
aoi_inds = [2, 3, 4, 5]
feature_type = 'buildings'
out_path = '/opt/data/research/ssl/spacenet.csv'
dfs = []

for aoi, aoi_ind in zip(aois, aoi_inds):
    label_dir = f's3://spacenet-dataset/spacenet/SN2_{feature_type}/train/AOI_{aoi_ind}_{aoi}/geojson_{feature_type}/'
    image_dir = f's3://spacenet-dataset/spacenet/SN2_{feature_type}/train/AOI_{aoi_ind}_{aoi}/PS-RGB/'
    image_fn_prefix = f'SN2_{feature_type}_train_AOI_{aoi_ind}_{aoi}_PS-RGB_img'
    label_fn_prefix = f'SN2_{feature_type}_train_AOI_{aoi_ind}_{aoi}_geojson_{feature_type}_img'

    label_paths = list_paths(label_dir, ext='.geojson')
    label_re = re.compile(r'.*{}(\d+)\.geojson'.format(label_fn_prefix))
    scene_ids = [label_re.match(label_path).group(1) for label_path in label_paths]

    aoi_info = []
    for scene_id in scene_ids:
        if aoi == 'Vegas' and scene_id == '1000':
            continue

        image_uri = join(image_dir, f'{image_fn_prefix}{scene_id}.tif')
        label_uri = join(label_dir, f'{label_fn_prefix}{scene_id}.geojson')
        scene_id = f'{aoi}_{scene_id}'
        aoi_info.append((scene_id.lower(), image_uri, label_uri))

    random.seed(1234)
    random.shuffle(aoi_info)

    scene_ids, image_uris, label_uris = list(zip(*aoi_info))
    train_ratio = 0.8
    train_sz = int(round(train_ratio * len(aoi_info)))
    split = ['train'] * train_sz + ['val'] * (len(aoi_info) - train_sz)
    df = pd.DataFrame(data={
        'aoi': aoi.lower(), 'split': split, 'scene_id': scene_ids, 'image_uri': image_uris,
        'label_uri': label_uris})
    dfs.append(df)
    print(f'{aoi}: {len(df)} scenes')

df = pd.concat(dfs)
make_dir(out_path, use_dirname=True)
df.to_csv(out_path, index=False)

# %%
