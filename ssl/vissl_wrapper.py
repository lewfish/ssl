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
# input: data uri, output uri, command to run
# download chips
# unzip them
# delete all the validation chips
# optional: download previous output, get checkpoint, setup sync thread
# run vissl
# upload to s3
