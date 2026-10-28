"""Legacy post-target adapter using the shared bounded GMM fitter."""
from ._gmm_bic import gmm_bic_model_1d


def gmm_bic_model_post(subtraction, data_pre, kmeans_labels, components,
                       max_workers=8, max_components=3, max_samples=50000,
                       bic_search=False, min_samples=1000):
    return gmm_bic_model_1d(subtraction + data_pre, data_pre, kmeans_labels, components,
                          max_workers=max_workers, max_components=max_components,
                          max_samples=max_samples, bic_search=bic_search, min_samples=min_samples)


gmm_bic_model_post.__modeling_target__ = "post"
