# -*- coding: utf-8 -*-
"""
LULC-stratified K-Means extractor.

Stratifies pixels by LULC class first, then runs FAISS K-Means within each
class on the remaining features.  This respects LULC as a nominal variable
and ensures cross-class dynamics never mix.
"""

import numpy as np


class LulcStratifiedKMeans:
    """Composite model: one FAISS K-Means wrapper per LULC class.

    Compatible with the detection pipeline — exposes ``.predict()``,
    ``.cluster_centers_``, ``.n_clusters``, and pickle support.
    """

    def __init__(self, models, offsets, n_clusters, lulc_band, lulc_values):
        self._models = models          # {lulc_value: FaissKMeansWrapper}
        self._offsets = offsets        # {lulc_value: int}
        self.n_clusters = n_clusters   # total across all classes
        self.lulc_band = lulc_band
        self.lulc_values = lulc_values  # sorted list, for iteration order

        # Build combined centroids for top-K search compatibility
        cents = []
        for lv in lulc_values:
            if lv in models:
                cents.append(models[lv].cluster_centers_)
        self.cluster_centers_ = (
            np.vstack(cents).astype(np.float32) if cents
            else np.empty((0, 0), dtype=np.float32)
        )

    def predict(self, X):
        """Assign each row to a global cluster label via LULC → sub-model."""
        lulc_col = X[:, self.lulc_band]
        labels = np.full(X.shape[0], -1, dtype=np.int32)
        for lv in self.lulc_values:
            model = self._models.get(lv)
            if model is None:
                continue
            mask = np.abs(lulc_col - lv) < 1e-4
            if mask.any():
                sub_labels = model.predict(X[mask])
                labels[mask] = sub_labels + self._offsets[lv]
        return labels

    def __getstate__(self):
        return (
            self._models,
            self._offsets,
            self.n_clusters,
            self.lulc_band,
            self.lulc_values,
        )

    def predict_neighbors(self, X, k):
        """Neighbors stay inside the observed LULC class; unknowns use -1."""
        count = min(k, self.n_clusters)
        labels = np.full((len(X), count), -1, dtype=np.int32)
        lulc = X[:, self.lulc_band]
        for value, model in self._models.items():
            mask = np.abs(lulc - value) < 1e-4
            if not mask.any():
                continue
            sub = model.predict_neighbors(X[mask], min(count, model.n_clusters))
            local = labels[mask]
            local[:, :sub.shape[1]] = sub + self._offsets[value]
            labels[mask] = local
        return labels

    def __setstate__(self, state):
        self._models, self._offsets, self.n_clusters, self.lulc_band, self.lulc_values = state
        # Rebuild cluster_centers_
        cents = []
        for lv in self.lulc_values:
            if lv in self._models:
                cents.append(self._models[lv].cluster_centers_)
        self.cluster_centers_ = (
            np.vstack(cents).astype(np.float32) if cents
            else np.empty((0, 0), dtype=np.float32)
        )


def lulc_stratified_kmeans_extractor(
    base_data: np.ndarray,
    component_number: int = 50,
    max_iter: int = 300,
    gpu: bool = False,
    lulc_band: int = -1,
    **kwargs,
):
    """Stratify by LULC class, then K-Means within each class.

    Parameters
    ----------
    base_data : np.ndarray (N, D)
        Feature matrix.  By default the last column is LULC.
    component_number : int
        Total clusters distributed proportionally across LULC classes.
    max_iter : int
        FAISS K-Means iterations.
    gpu : bool
        Use FAISS GPU backend.
    lulc_band : int
        Column index of the LULC band (default -1 = last).

    Returns
    -------
    (labels, model) : (np.ndarray, LulcStratifiedKMeans)
    """
    import faiss
    from ._faiss import FaissKMeansWrapper

    n_samples, n_features = base_data.shape
    lulc_col = base_data[:, lulc_band]

    # Identify discrete LULC classes from normalised values
    lulc_unique = np.unique(np.round(lulc_col, 4))
    lulc_classes = sorted(lulc_unique)

    models = {}
    offsets = {}
    all_labels = np.full(n_samples, -1, dtype=np.int32)
    offset = 0

    for lv in lulc_classes:
        mask = np.abs(lulc_col - lv) < 1e-4
        n_class = mask.sum()
        if n_class == 0:
            continue

        # Proportional allocation, at least 1 cluster per class
        n_clu = max(1, int(round(component_number * n_class / n_samples)))
        n_clu = min(n_clu, n_class)

        sub_data = base_data[mask].astype(np.float32)

        kmeans = faiss.Kmeans(
            d=n_features, k=n_clu, niter=max_iter, nredo=1,
            gpu=gpu, seed=42, verbose=False,
        )
        kmeans.train(sub_data)

        wrapper = FaissKMeansWrapper()
        wrapper.n_clusters = n_clu
        wrapper.cluster_centers_ = np.asarray(kmeans.centroids, dtype=np.float32)
        wrapper._build_index()

        models[lv] = wrapper
        offsets[lv] = offset

        _, sub_labels = wrapper._index.search(sub_data, 1)
        all_labels[mask] = sub_labels.ravel() + offset
        offset += n_clu

    composite = LulcStratifiedKMeans(models, offsets, offset, lulc_band, lulc_classes)
    return all_labels, composite
