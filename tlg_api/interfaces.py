"""Small structural protocols for independently packaged algorithm extensions."""
from typing import Protocol
import numpy as np


class ComponentExtractor(Protocol):
    n_clusters: int

    def predict(self, values: np.ndarray) -> np.ndarray: ...

    def predict_neighbors(self, values: np.ndarray, k: int) -> np.ndarray:
        """Return global component IDs; -1 indicates no eligible component."""
        ...


class StatisticalModel(Protocol):
    def fit(self, values: np.ndarray, labels: np.ndarray, components) -> "StatisticalModel": ...

    def score(self, values: np.ndarray, labels: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Return (pixels, bands) scores; missing models return NaN."""
        ...
