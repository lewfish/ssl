# %%
from IPython import get_ipython
from IPython.display import Image, display

from os.path import join, dirname, basename
from io import StringIO
import tempfile

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from rastervision.pipeline.file_system import (
    make_dir, list_paths, file_to_json, file_to_str, download_if_needed)

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", None)

# %%
from collections import OrderedDict
from vissl_experiments.vissl_wrapper import extract_backbone

ckpt_path = '/opt/data/nih-histo/models/nih-moco.torch'
out_path = '/opt/data/nih-histo/models/nih-moco-backbone.torch'
extract_backbone(ckpt_path, out_path)

# %%
