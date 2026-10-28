"""Validated training pipeline with bounded, reproducible sampling."""
import numpy as np
from ._io import (validate_pair, flat_block, iter_windows, read_context_stack, band_names)
from .algorithms import resolve_algo
from .algorithms._registry import normalized_config
from ._bundle import BUNDLE_VERSION


def train(pre_path, post_path, config=None, max_workers=8, block_size=0,
          verbose=True, context=None, context_ranges=None, context_normalize=True,
          use_pre_for_clustering=True, max_training_samples=None, random_state=42,
          feature_names=None, context_types=None, **simple_kwargs):
    """Train from aligned rasters.

    block_size > 0 uses uniform random-priority sampling, bounded by
    max_training_samples (default 250000 for block mode). It is sampled batch
    training, not incremental fitting. Context rasters are read by window too.
    Full mode preserves all valid pixels unless an explicit sample cap is set.
    """
    if config is not None and simple_kwargs:
        raise ValueError("Use either config or shorthand algorithm arguments")
    config = normalized_config(config if config is not None else _simple_kwargs_to_config(simple_kwargs))
    if not isinstance(block_size, int) or block_size < 0:
        raise ValueError("block_size must be a non-negative integer")
    if not isinstance(max_workers, int) or max_workers == 0:
        raise ValueError("max_workers must be a non-zero integer")
    if block_size and max_training_samples is None:
        max_training_samples = 250000
    if max_training_samples is not None and (not isinstance(max_training_samples, int) or max_training_samples < 2):
        raise ValueError("max_training_samples must be an integer >= 2")
    if not use_pre_for_clustering and not context:
        raise ValueError("use_pre_for_clustering=False requires context")
    info = validate_pair(pre_path, post_path)
    h, w, c, _, _ = info
    names = list(feature_names) if feature_names is not None else band_names(pre_path)
    if names is not None and (len(names) != c or len(set(names)) != c):
        raise ValueError("feature_names must contain one unique name per spectral band")
    rng = np.random.default_rng(random_state)
    kept, priorities = None, np.empty(0)
    total_valid = 0
    context_meta = {}
    cache = {}
    for x, y, bw, bh in iter_windows(h, w, block_size or max(h, w)):
        pre = flat_block(pre_path, x, y, bw, bh, cache)
        post = flat_block(post_path, x, y, bw, bh, cache)
        ctx, context_meta = read_context_stack(context, h, w, reference=info,
                                               window=(x, y, bw, bh), ds_cache=cache)
        values = np.hstack([pre, post, ctx]) if ctx is not None else np.hstack([pre, post])
        values = values[np.isfinite(values).all(axis=1)]
        total_valid += len(values)
        keys = rng.random(len(values))
        if kept is not None:
            values = np.vstack([kept, values])
            keys = np.concatenate([priorities, keys])
        if max_training_samples is not None and len(values) > max_training_samples:
            take = np.argpartition(keys, max_training_samples - 1)[:max_training_samples]
            values, keys = values[take], keys[take]
        kept, priorities = values, keys
    if kept is None or not len(kept):
        raise ValueError("No valid pixels remain after raster and context masking")
    pre, post = kept[:, :c], kept[:, c:2*c]
    ctx = kept[:, 2*c:]
    if ctx.shape[1]:
        # Preserve nominal LULC IDs even when the caller uses the old default
        # context_normalize=True. Clipping an unseen class would turn it into a
        # known endpoint class and silently defeat stratification.
        if "lulc_stratified_kmeans_extractor" in config["extractor"]:
            context_types = dict(context_types or {})
            params = config["extractor"]["lulc_stratified_kmeans_extractor"]
            feature_count = ctx.shape[1] + (c if use_pre_for_clustering else 0)
            index = params.get("lulc_band", -1)
            if not isinstance(index, int) or not -feature_count <= index < feature_count:
                raise ValueError("lulc_band is outside the feature schema")
            index %= feature_count
            index -= c if use_pre_for_clustering else 0
            offset = 0
            for name, count in zip(context_meta["context_names"], context_meta["context_band_counts"]):
                if offset <= index < offset + count:
                    if context_types.get(name, "categorical") != "categorical":
                        raise ValueError("LULC stratification requires categorical context")
                    context_types[name] = "categorical"
                    break
                offset += count
        ctx, norm_meta = _normalize_context(ctx, context_meta["context_names"],
                                            context_meta["context_band_counts"],
                                            context_ranges, context_normalize, context_types)
        context_meta.update(norm_meta)
        features = np.hstack([pre, ctx]) if use_pre_for_clustering else ctx
    else:
        if not use_pre_for_clustering:
            raise ValueError("Context contains no features")
        features = pre
    modeler, mod_params = resolve_algo("modeler", config)
    target = getattr(modeler, "__modeling_target__", "delta")
    if target not in ("delta", "post"):
        raise ValueError("modeling_target must be delta or post")
    ds_func, ds_params = resolve_algo("dynamic_system", config)
    dynamics = np.asarray(ds_func(pre, post, **ds_params), dtype=np.float32)
    if dynamics.shape != pre.shape:
        raise ValueError("Dynamic system must return shape (pixels, source_bands)")
    finite = np.isfinite(post if target == "post" else dynamics).all(axis=1)
    pre, post, dynamics, features = pre[finite], post[finite], dynamics[finite], features[finite]
    if not len(pre):
        raise ValueError("Dynamic system produced no finite training values")
    ext_func, ext_params = resolve_algo("extractor", config)
    if ext_params.get("component_number", 1) > len(pre):
        raise ValueError("component_number exceeds available training samples")
    labels, extractor = ext_func(features, **ext_params)
    labels = np.asarray(labels)
    n_clusters = int(getattr(extractor, "n_clusters", labels.max() + 1))
    if labels.shape != (len(pre),) or n_clusters < 1 or (labels < 0).any() or (labels >= n_clusters).any():
        raise ValueError("Extractor must return valid dense component labels")
    modeler_name = next(iter(config["modeler"]))
    result = {}
    if getattr(modeler, "__tlg_fit_score__", False):
        score_model = modeler(**mod_params)
        if not callable(getattr(score_model, "fit", None)) or not callable(getattr(score_model, "score", None)):
            raise TypeError("Model factory must return an object implementing fit and score")
        score_model.fit(post if target == "post" else dynamics, labels, range(n_clusters))
        result["score_model"] = score_model
    else:
        # Legacy post modelers reconstruct post = pre + argument.
        training_values = post - pre if target == "post" else dynamics
        prior = modeler(training_values, pre, labels, range(n_clusters),
                        max_workers=max_workers, **mod_params)
        result["gmm_list"] = [prior.get(i) for i in range(n_clusters)]
    context_meta.update(source_bands=c, use_pre_for_clustering=use_pre_for_clustering,
                        modeling_target=target)
    result.update(
        schema_version=BUNDLE_VERSION, model=extractor, algorithm_config=config,
        modeler_name=modeler_name, context_meta=context_meta,
        feature_schema={"source_bands": c, "spectral_names": names,
                        "context_names": context_meta.get("context_names", []),
                        "context_band_counts": context_meta.get("context_band_counts", [])},
        training_meta={"valid_pixels": total_valid, "training_samples": len(pre),
                       "max_training_samples": max_training_samples, "block_size": block_size,
                       "random_state": random_state, "sampling": "uniform_random_priority",
                       "pre_source": str(pre_path), "post_source": str(post_path)})
    if verbose:
        print(f"[TLG Train] fitted {n_clusters} components from {len(pre)} / {total_valid} valid pixels")
    return result


def _normalize_context(ctx_stack, names, band_counts, context_ranges, normalize,
                       context_types=None):
    ctx_stack = ctx_stack.copy()
    types = context_types or {}
    if set(types) - set(names) or (context_ranges and set(context_ranges) - set(names)):
        raise ValueError("Context normalization contains unknown feature names")
    if any(value not in ("continuous", "categorical") for value in types.values()):
        raise ValueError("Context types must be continuous or categorical")
    meta = {"normalize": normalize, "context_ranges": {}, "context_mins": [],
            "context_maxs": [], "context_types": {}, "context_normalized_bands": []}
    col = 0
    for name, count in zip(names, band_counts):
        kind = types.get(name, "continuous")
        meta["context_types"][name] = kind
        user_range = (context_ranges or {}).get(name)
        if user_range is not None:
            if len(user_range) != 2 or not np.isfinite(user_range).all() or user_range[1] <= user_range[0]:
                raise ValueError(f"Invalid normalization range for {name}")
        meta["context_ranges"][name] = user_range
        for index in range(col, col + count):
            lo, hi = user_range or (float(ctx_stack[:, index].min()), float(ctx_stack[:, index].max()))
            enabled = bool(normalize and kind == "continuous")
            meta["context_mins"].append(lo)
            meta["context_maxs"].append(hi)
            meta["context_normalized_bands"].append(enabled)
            if enabled:
                ctx_stack[:, index] = (ctx_stack[:, index] - lo) / (hi - lo) if hi > lo else 0
                ctx_stack[:, index] = np.clip(ctx_stack[:, index], 0, 1)
        col += count
    return ctx_stack, meta


_SIMPLE_TO_CONFIG_MAP = {
    "component_number": ("extractor", "faiss_kmeans_extractor"),
    "max_iter": ("extractor", "faiss_kmeans_extractor"),
    "gpu": ("extractor", "faiss_kmeans_extractor"),
    "max_components": ("modeler", "gmm_bic_model_1d"),
    "max_samples": ("modeler", "gmm_bic_model_1d")}


def _simple_kwargs_to_config(kwargs):
    config = {}
    for key, value in kwargs.items():
        if key not in _SIMPLE_TO_CONFIG_MAP:
            raise TypeError(f"train() got unexpected keyword argument '{key}'")
        stage, algo = _SIMPLE_TO_CONFIG_MAP[key]
        config.setdefault(stage, {}).setdefault(algo, {})[key] = value
    return config or None
