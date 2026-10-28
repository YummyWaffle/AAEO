# -*- coding: utf-8 -*-
"""
Mini-Batch K-Means extractor (default).
"""

import numpy as np


def mb_kmeans_extractor(
    base_data: np.ndarray,
    component_number: int = 50,
    max_iter: int = 100,
    batch_size: int = 100000,
    **kwargs,
):
    """
    Cluster pre-event pixels into spectral components via Mini-Batch K-Means.

    Returns (labels, model) where labels is a 1D array of cluster assignments
    and model is the fitted MiniBatchKMeans object.
    """
    from sklearn.cluster import MiniBatchKMeans
    extractor = MiniBatchKMeans(
        n_clusters=component_number,
        init='k-means++',
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=42,
        verbose=0,
    )
    labels = extractor.fit_predict(base_data)
    return labels, extractor
