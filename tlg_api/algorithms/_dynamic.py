# -*- coding: utf-8 -*-
"""
Dynamic system — computes the difference (or ratio, etc.) between two images.
"""

import numpy as np


def difference_dynamic(pre_image: np.ndarray, post_image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute per-pixel difference between post- and pre-event images."""
    return np.subtract(post_image, pre_image)


difference_dynamic.__window_safe__ = True
