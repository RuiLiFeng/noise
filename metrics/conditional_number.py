# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class CondN(metric_base.MetricBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        Gs_clone = Gs.clone()
        cond_list = []

        for name, var in Gs_clone.components.synthesis.trainables.items():
            w = var.eval()
            cond_list.append(np.linalg.cond(w))
        cond = np.mean(cond_list)
        self._report_result(np.real(cond))

#----------------------------------------------------------------------------
