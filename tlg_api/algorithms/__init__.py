# -*- coding: utf-8 -*-
"""
TLG algorithms package — decoupled per-stage algorithm registry.

Each stage (dynamic_system, extractor, modeler, merger) lives in its own
subpackage.  To add a custom algorithm, either:

1. Register at runtime::

       from tlg_api import register_extractor
       register_extractor("my_algo", my_func)

2. Or drop a new file into the appropriate subpackage and add an import
   line to its ``__init__.py`` (same pattern as the historic code).
"""

# Registry infrastructure
from ._registry import (
    register_dynamic_system,
    register_extractor,
    register_modeler,
    register_merger,
    list_registered,
    resolve_algo,
)

# Import subpackages to trigger built-in registrations
from . import _dynamic
from ._registry import register_dynamic_system as _reg_ds

_reg_ds("difference_dynamic", _dynamic.difference_dynamic)

from . import _extractors   # registers all extractors
from . import _modelers     # registers all modelers
from . import _mergers      # registers all mergers
