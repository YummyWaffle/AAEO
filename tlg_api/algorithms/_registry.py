# -*- coding: utf-8 -*-
"""
Algorithm registry infrastructure — dicts, register functions, and config resolution.

This is the only shared state across all algorithm modules.  Each algorithm
subpackage imports from here and calls the appropriate ``register_*`` function.
"""

# ---------------------------------------------------------------------------
# Algorithm registries (module-level dicts, like the historic code's __init__.py)
# ---------------------------------------------------------------------------

_DYNAMIC_SYSTEMS: dict = {}
_EXTRACTORS: dict = {}
_MODELERS: dict = {}
_MERGERS: dict = {}
_SCORERS: dict = {}


def _validate_registration(registry, name, func, replace):
    """Validate the complete request before changing any registry state."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Algorithm name must be a non-empty string")
    if not callable(func):
        raise TypeError("Registered algorithm must be callable")
    if name in registry and not replace:
        raise ValueError(f"Algorithm '{name}' is already registered; use replace=True explicitly")


def _register(registry, name, func, replace):
    _validate_registration(registry, name, func, replace)
    registry[name] = func
    return func


def register_dynamic_system(name: str, func, *, replace=False):
    """Register (pre, post, **params) -> dynamics; duplicates require replace=True."""
    return _register(_DYNAMIC_SYSTEMS, name, func, replace)


def register_extractor(name: str, func, *, replace=False):
    """Register (data, **params) -> (labels, model); duplicates require replace=True."""
    return _register(_EXTRACTORS, name, func, replace)


def register_modeler(name: str, func, scorer=None, *, replace=False):
    """Register a legacy training function, or a factory returning fit/score.

    Optional scorer(models, values, sigma=...) must return (N, bands).
    New model factories expose __tlg_fit_score__ = True and return an object
    implementing fit(values, labels, components) and score(values, labels, sigma).
    Duplicates require replace=True. Replacing without a scorer clears the old one.
    """
    _validate_registration(_MODELERS, name, func, replace)
    if scorer is not None and not callable(scorer):
        raise TypeError("Registered model scorer must be callable")
    _MODELERS[name] = func
    if scorer is not None:
        _SCORERS[name] = scorer
    else:
        _SCORERS.pop(name, None)
    return func


def get_scorer(name):
    return _SCORERS.get(name)


def register_merger(name: str, func, *, replace=False):
    """Register (atomic_priors list) -> merged dict; duplicates require replace=True."""
    return _register(_MERGERS, name, func, replace)


def list_registered():
    """Return a dict showing all registered algorithm names per stage."""
    return {
        "dynamic_system": list(_DYNAMIC_SYSTEMS),
        "extractor": list(_EXTRACTORS),
        "modeler": list(_MODELERS),
        "merger": list(_MERGERS),
    }


# ---------------------------------------------------------------------------
# Config → algo resolution (like historic code's TlgEngine._get_algo_config)
# ---------------------------------------------------------------------------

_STAGE_REGISTRY = {
    "dynamic_system": _DYNAMIC_SYSTEMS,
    "extractor": _EXTRACTORS,
    "modeler": _MODELERS,
    "merger": _MERGERS,
}

_DEFAULT_ALGOS = {
    "dynamic_system": "difference_dynamic",
    "extractor": "faiss_kmeans_extractor",
    "modeler": "gmm_bic_model_1d",
    "merger": "none",
}

_DEFAULT_PARAMS = {
    "extractor": {"component_number": 50, "max_iter": 300, "gpu": False},
    "modeler": {"max_components": 3, "max_samples": 50000},
}


def resolve_algo(stage: str, config: dict = None):
    """
    Given a stage name and an optional config dict, return (func, params).

    If *config* is ``None`` or doesn't mention *stage*, the default
    registered algorithm and its default params are returned.

    Config format (matches historic algorithm_config.json)::

        {
          "extractor": {
            "mini_batch_k_means_extractor": {"component_number": 50}
          }
        }
    """
    import inspect
    if stage not in _STAGE_REGISTRY:
        raise ValueError(f"Unknown algorithm stage: {stage}")
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("Algorithm config must be a dictionary")
        unknown = set(config) - set(_STAGE_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown algorithm stages: {sorted(unknown)}")
    registry = _STAGE_REGISTRY[stage]

    if config and stage in config:
        stage_cfg = config[stage]
        if not isinstance(stage_cfg, dict) or len(stage_cfg) != 1:
            raise ValueError(f"Stage '{stage}' must select exactly one algorithm")
        algo_name = next(iter(stage_cfg))
        algo_params = {} if stage_cfg[algo_name] is None else stage_cfg[algo_name]
    else:
        algo_name = _DEFAULT_ALGOS[stage]
        algo_params = _DEFAULT_PARAMS.get(stage, {})

    func = registry.get(algo_name)
    if func is None:
        raise ValueError(
            f"Unknown {stage} algorithm '{algo_name}'. "
            f"Registered: {list(registry)}"
        )
    if not isinstance(algo_params, dict):
        raise TypeError(f"Parameters for '{algo_name}' must be a dictionary")
    signature = inspect.signature(func)
    allowed = {key for key, p in signature.parameters.items()
               if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    allowed.update(getattr(func, "__parameter_names__", ()))
    reserved = {
        "dynamic_system": list(signature.parameters)[:2],
        "extractor": list(signature.parameters)[:1],
        "modeler": [] if getattr(func, "__tlg_fit_score__", False) else list(signature.parameters)[:4] + ["max_workers"],
        "merger": list(signature.parameters)[:1],
    }[stage]
    allowed.difference_update(reserved)
    unknown = set(algo_params) - allowed
    if unknown:
        raise ValueError(f"Unknown parameters for '{algo_name}': {sorted(unknown)}")
    return func, dict(algo_params)


def normalized_config(config=None):
    """Resolve and validate all stages before raster reads or model fitting."""
    result = {}
    for stage in _STAGE_REGISTRY:
        _, params = resolve_algo(stage, config)
        name = next(iter(config[stage])) if config and stage in config else _DEFAULT_ALGOS[stage]
        result[stage] = {name: params}
    if next(iter(result["merger"])) != "none":
        raise ValueError("train() has one model; merger is reserved for explicit library merging")
    return result
