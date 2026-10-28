# -*- coding: utf-8 -*-
"""
TLG API — Temporal Learning of Gaussian Models
===============================================

A pure Python library for remote sensing change detection using the
Three-stage Learning of Gaussian (TLG) algorithm.

Quickstart
----------
    from tlg_api import train, detect, save_bundle, save_detection, anomaly_preview

    # Train a model
    bundle = train("pre_event.tif", "post_event.tif")
    save_bundle(bundle, "model.pkg")

    # Detect anomalies (max-only by default)
    result = detect("pre_event.tif", "post_event.tif", "model.pkg")
    anomaly_map = result["anomaly_map"]

    # Save result and preview
    save_detection(result, "anomaly.tif")
    anomaly_preview(anomaly_map, "preview.jpg", background_path="pre_event.tif")

Custom algorithms
-----------------
    from tlg_api import register_extractor, list_registered

    def my_clustering(data, **params):
        ...
        return labels, model

    register_extractor("my_clustering", my_clustering)

    bundle = train("pre.tif", "post.tif", config={
        "extractor": {"my_clustering": {"n_clusters": 80}},
    })
"""

from ._train import train
from ._detect import detect, detect_to_file, save_detection
from ._bundle import save_bundle, load_bundle
from .algorithms import (
    register_dynamic_system,
    register_extractor,
    register_modeler,
    register_merger,
    list_registered,
)


def anomaly_preview(*args, **kwargs):
    """Load optional plotting dependencies only when creating a preview."""
    from ._viz import anomaly_preview as render
    return render(*args, **kwargs)


__all__ = [
    "train",
    "detect",
    "detect_to_file",
    "save_detection",
    "save_bundle",
    "load_bundle",
    "anomaly_preview",
    "register_dynamic_system",
    "register_extractor",
    "register_modeler",
    "register_merger",
    "list_registered",
]
