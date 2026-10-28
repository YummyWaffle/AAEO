# -*- coding: utf-8 -*-
"""
Post-hoc mergers — combine atomic priors from multiple sources.
"""

from .._registry import register_merger
from ._noop import _noop_merger

register_merger("none", _noop_merger)
