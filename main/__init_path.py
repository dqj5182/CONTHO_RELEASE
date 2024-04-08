from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import torch
import numpy as np


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

data_path = osp.join(this_dir, '..', 'data')
add_path(data_path)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

util_path = osp.join(this_dir, '..', 'lib', 'utils')
add_path(util_path)