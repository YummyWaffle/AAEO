# -*- coding: utf-8 -*-
"""
FAISS K-Means extractor — GPU-accelerated, much faster than sklearn on large data.
"""

import numpy as np


class FaissKMeansWrapper:
    """Adds a sklearn-compatible ``.predict()`` to a FAISS KMeans object.

    Implements pickle support so bundles can be saved with ``save_bundle()``.
    """

    def __init__(self, faiss_kmeans=None):
        self.faiss_kmeans = faiss_kmeans
        if faiss_kmeans is not None:
            self.n_clusters = faiss_kmeans.k
            self.cluster_centers_ = np.asarray(faiss_kmeans.centroids)
            self._build_index()

    def _build_index(self):
        import faiss
        d = self.cluster_centers_.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(self.cluster_centers_)

    def predict(self, X):
        X_f32 = np.asarray(X, dtype=np.float32)
        _, labels = self._index.search(X_f32, 1)
        return labels.ravel()

    def predict_neighbors(self, X, k):
        """Exact nearest neighbors within this extractor's own index."""
        _, labels = self._index.search(np.asarray(X, dtype=np.float32), min(k, self.n_clusters))
        return labels

    def __getstate__(self):
        return (self.n_clusters, self.cluster_centers_)

    def __setstate__(self, state):
        self.n_clusters, self.cluster_centers_ = state
        self.faiss_kmeans = None
        self._build_index()


def faiss_kmeans_extractor(
    base_data: np.ndarray,
    component_number: int = 50,
    max_iter: int = 300,
    nredo: int = 1,
    gpu: bool = False,
    **kwargs,
):
    """
    Cluster pre-event pixels via FAISS K-Means.

    Returns (labels, model) where model is a ``FaissKMeansWrapper``.
    FAISS is highly optimized and can be orders of magnitude faster than
    sklearn for large datasets.  Set ``gpu=True`` for GPU acceleration.
    """
    import faiss

    data_f32 = np.asarray(base_data, dtype=np.float32)
    d = data_f32.shape[1]

    kmeans = faiss.Kmeans(
        d=d, k=component_number,
        niter=max_iter,
        nredo=nredo,
        seed=42,
        gpu=gpu,
        verbose=False,
    )
    kmeans.train(data_f32)

    wrapper = FaissKMeansWrapper(kmeans)
    labels = wrapper.predict(data_f32)
    return labels, wrapper
