"""Versioned model bundles. Load only trusted joblib files."""
import os
from pathlib import Path
import warnings
import uuid
import joblib

BUNDLE_VERSION = 2


def normalize_bundle(bundle):
    if isinstance(bundle, (str, os.PathLike)):
        bundle = joblib.load(bundle)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid TLG model bundle")
    result = dict(bundle)
    version = result.get("schema_version", 1)
    if version not in (1, BUNDLE_VERSION):
        raise ValueError(f"Unsupported model schema_version: {version}")
    config = result.get("algorithm_config")
    if config is not None and not isinstance(config, dict):
        raise ValueError("Bundle algorithm_config must be a dictionary")
    if version == BUNDLE_VERSION and (
            not isinstance(config, dict) or "dynamic_system" not in config):
        raise ValueError("Version 2 bundles must include explicit dynamic_system configuration")
    if version == 1:
        config = dict(config or {})
        if "dynamic_system" not in config:
            warnings.warn("Legacy bundle: assuming difference_dynamic; custom legacy transforms "
                          "cannot be inferred and require explicit metadata migration.",
                          UserWarning, stacklevel=2)
            config["dynamic_system"] = {"difference_dynamic": {}}
            result["legacy_assumptions"] = ["difference_dynamic"]
        else:
            warnings.warn("Legacy bundle adapted using supplied dynamic configuration.",
                          UserWarning, stacklevel=2)
        result["algorithm_config"] = config
        result.setdefault("modeler_name", "gmm_bic_model_1d")
        result["schema_version"] = BUNDLE_VERSION
    dynamic = config["dynamic_system"]
    if not isinstance(dynamic, dict) or len(dynamic) != 1:
        raise ValueError("Bundle dynamic_system must select exactly one algorithm")
    name, parameters = next(iter(dynamic.items()))
    if not isinstance(name, str) or not name or not isinstance(parameters, dict):
        raise ValueError("Bundle dynamic_system needs an algorithm name and parameter dictionary")
    for key in ("context_meta", "feature_schema", "training_meta"):
        if key in result and not isinstance(result[key], dict):
            raise ValueError(f"Bundle {key} must be a dictionary")
    if "gmm_list" not in result and "score_model" not in result:
        raise ValueError("Bundle must contain gmm_list or a fit/score model")
    return result


def save_bundle(bundle, path):
    """Atomically save a trusted Python model; does not mutate the input bundle."""
    normalized = normalize_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        joblib.dump(normalized, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bundle(path):
    """Load a trusted .pkg file, validate version, and adapt legacy metadata."""
    return normalize_bundle(path)
