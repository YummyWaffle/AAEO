"""Small non-GMM reference implementation of the fit/score extension contract."""
import numpy as np


class RobustDeviationModel:
    """Per-component median/MAD envelope; scores have the GMM score range [0, 1]."""
    __tlg_fit_score__ = True

    def __init__(self, min_samples=10, scale_floor=1e-6):
        if not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be positive and finite")
        self.min_samples = min_samples
        self.scale_floor = scale_floor
        self.parameters = {}

    def fit(self, values, labels, components):
        self.parameters = {}
        for component in components:
            subset = values[labels == component]
            if len(subset) < self.min_samples:
                continue
            median = np.median(subset, axis=0)
            scale = np.maximum(np.median(np.abs(subset - median), axis=0) * 1.4826, self.scale_floor)
            self.parameters[component] = median, scale
        return self

    def score(self, values, labels, sigma=2.0):
        result = np.full(values.shape, np.nan, dtype=np.float32)
        for component in np.unique(labels):
            if component not in self.parameters:
                continue
            median, scale = self.parameters[component]
            mask = labels == component
            distance = np.abs(values[mask] - median)
            result[mask] = np.maximum(0, 1 - sigma * scale / (distance + 1e-12))
        return result
