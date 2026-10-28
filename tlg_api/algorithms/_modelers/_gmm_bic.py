# -*- coding: utf-8 -*-
"""
GMM BIC modeler — 1-D Gaussian Mixture Model per component per band.
"""

import numpy as np
from joblib import Parallel, delayed


def _gmm_unit(comp_id, sub_data, max_components, max_samples, bic_search, min_samples=1000):
    """Fit 1-D GMM for one component, band by band (runs in subprocess)."""
    count = sub_data.shape[0]
    from sklearn.mixture import GaussianMixture
    if count < min_samples:
        return comp_id, None

    if count > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(count, max_samples, replace=False)
        modeling_data = sub_data[idx]
    else:
        modeling_data = sub_data

    n_bands = sub_data.shape[1]
    band_models_list = []

    for b in range(n_bands):
        band_data = modeling_data[:, b].reshape(-1, 1)
        n_fit = min(max_components, band_data.shape[0])

        if bic_search and n_fit > 1:
            # Try 1 .. n_fit components, pick best BIC.
            best_gmm = None
            lowest_bic = np.inf
            for n in range(1, n_fit + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=n,
                        covariance_type='spherical',
                        max_iter=50,
                        random_state=42,
                    )
                    gmm.fit(band_data)
                    bic = gmm.bic(band_data)
                    if bic < lowest_bic:
                        lowest_bic = bic
                        best_gmm = gmm
                except Exception:
                    continue
        else:
            # Fit directly with n_fit components (much faster).
            try:
                best_gmm = GaussianMixture(
                    n_components=n_fit,
                    covariance_type='spherical',
                    max_iter=50,
                    random_state=42,
                )
                best_gmm.fit(band_data)
            except Exception:
                best_gmm = None

        if best_gmm is not None:
            m = best_gmm.means_.flatten()
            v = np.sqrt(best_gmm.covariances_).flatten()
            w = best_gmm.weights_.flatten()

            valid_idx = w >= 0.05
            if valid_idx.any():
                m, v, w = m[valid_idx], v[valid_idx], w[valid_idx]
                w /= w.sum()

            band_models_list.append((m, v, w))
        else:
            band_models_list.append(None)

    return comp_id, band_models_list


def gmm_bic_model_1d(
    subtraction: np.ndarray,
    data_pre: np.ndarray,
    kmeans_labels: np.ndarray,
    components: range,
    max_workers: int = 8,
    max_components: int = 3,
    max_samples: int = 50000,
    bic_search: bool = False,
    min_samples: int = 1000,
    **kwargs,
):
    """
    Parallel 1-D GMM fitting for each component.

    Returns {comp_id: [band_0_model, band_1_model, ...], ...}
    where each band model is (means, stds, weights) or None.

    Parameters
    ----------
    bic_search : bool
        If True, try 1..max_components and pick the best via BIC (3× slower).
        Default False fits max_components directly — weight pruning still
        discards negligible components.
    """
    if min_samples < 2 or max_samples < min_samples or max_components < 1:
        raise ValueError("Require max_samples >= min_samples >= 2 and max_components >= 1")
    def tasks():
        rng = np.random.default_rng(42)
        for comp_id in components:
            indices = np.flatnonzero(kmeans_labels == comp_id)
            if len(indices) > max_samples:
                indices = rng.choice(indices, max_samples, replace=False)
            yield delayed(_gmm_unit)(
                comp_id, subtraction[indices], max_components, max_samples, bic_search, min_samples)

    results = Parallel(
        n_jobs=max_workers,
        backend='threading',
        pre_dispatch='2*n_jobs',
        verbose=0,
    )(tasks())

    prior_knowledge_base = {}
    for cid, value in results:
        if value is not None:
            prior_knowledge_base[cid] = value

    return prior_knowledge_base
