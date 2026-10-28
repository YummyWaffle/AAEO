# -*- coding: utf-8 -*-
"""
Component extractors — clustering algorithms for pre-event pixels.

To add a new extractor:
1. Create ``_your_algo.py`` in this directory.
2. Add two lines below:

   from ._your_algo import your_extractor
   register_extractor("your_extractor", your_extractor)
"""

from .._registry import register_extractor

from ._faiss import faiss_kmeans_extractor
from ._mb_kmeans import mb_kmeans_extractor
from ._lulc_stratified import lulc_stratified_kmeans_extractor

register_extractor("faiss_kmeans_extractor", faiss_kmeans_extractor)
register_extractor("mini_batch_k_means_extractor", mb_kmeans_extractor)
register_extractor("lulc_stratified_kmeans_extractor", lulc_stratified_kmeans_extractor)
