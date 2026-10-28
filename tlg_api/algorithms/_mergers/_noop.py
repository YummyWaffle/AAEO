# -*- coding: utf-8 -*-
"""
No-op merger — passes through atomic priors unchanged.
"""


def _noop_merger(atomic_priors, **kwargs):
    """Default no-op merger — returns the union of all priors unchanged."""
    return {}
