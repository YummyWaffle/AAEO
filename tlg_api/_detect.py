"""Inference shared by full-image and bounded window execution."""
import os
from pathlib import Path
import uuid
from contextlib import suppress
import numpy as np
from scipy import ndimage
from osgeo import gdal
from ._io import (validate_pair, flat_block, read_context_stack, write_geotiff,
                  iter_windows, band_names, _named_context)
from ._bundle import normalize_bundle
from .algorithms import resolve_algo
from .algorithms._registry import get_scorer


def save_detection(result, path, descriptions=None):
    data = result.get("band_scores", result["anomaly_map"][:, :, None])
    write_geotiff(path, data, result["geo_transform"], result["projection"], descriptions)


def _worker_calc_score(cat_id, diff_subset, models, sigma_ratio):
    """Legacy GMM adapter: absent models produce NaN, never 'normal' scores."""
    scores = np.full(diff_subset.shape, np.nan, dtype=np.float32)
    if models is None:
        return cat_id, scores
    if len(models) != diff_subset.shape[1]:
        raise ValueError("Model band count differs from the target values")
    for b, model in enumerate(models):
        if model is None:
            continue
        means, stds, _ = model
        band_score = np.full(len(diff_subset), np.inf, dtype=np.float32)
        for mean, std in zip(means, stds):
            if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
                continue
            distance = np.abs(diff_subset[:, b] - mean)
            value = np.maximum(0, 1 - sigma_ratio * std / (distance + 1e-12))
            band_score = np.minimum(band_score, value)
        band_score[~np.isfinite(band_score)] = np.nan
        scores[:, b] = band_score
    return cat_id, scores


def _compute_topk(model, values, count):
    if hasattr(model, "predict_neighbors"):
        result = np.asarray(model.predict_neighbors(values, count), dtype=np.int32)
    elif count == 1:
        result = np.asarray(model.predict(values), dtype=np.int32).reshape(-1, 1)
    else:
        # Only plain sklearn KMeans gets this legacy adapter. Custom extractors
        # must own their neighbor semantics (e.g. land-cover strata).
        from sklearn.cluster import KMeans, MiniBatchKMeans
        if not isinstance(model, (KMeans, MiniBatchKMeans)):
            raise TypeError("k_neighbors > 1 requires extractor.predict_neighbors(X, k)")
        count = min(count, model.n_clusters)
        result = np.empty((len(values), count), dtype=np.int32)
        for start in range(0, len(values), 4096):
            distances = model.transform(values[start:start + 4096])
            result[start:start + 4096] = np.argsort(distances, axis=1)[:, :count]
    if result.ndim != 2 or result.shape[0] != len(values) or result.shape[1] < 1:
        raise ValueError("predict_neighbors must return shape (pixels, neighbors)")
    return result


def _check_options(sigma, filter_size, output_mode, k_neighbors):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if not isinstance(filter_size, int) or filter_size < 1 or filter_size % 2 != 1:
        raise ValueError("filter_size must be a positive odd integer")
    if output_mode not in ("max", "full"):
        raise ValueError("output_mode must be 'max' or 'full'")
    if not isinstance(k_neighbors, int) or k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")


def _check_schema(pkg, pre_path, info, feature_names=None):
    schema = pkg.get("feature_schema", {})
    expected_count = schema.get("source_bands", pkg.get("context_meta", {}).get("source_bands"))
    if expected_count is not None and expected_count != info[2]:
        raise ValueError("Spectral band count differs from the model feature schema")
    expected_names = schema.get("spectral_names")
    actual_names = list(feature_names) if feature_names is not None else band_names(pre_path)
    if expected_names is not None and actual_names != expected_names:
        raise ValueError("Spectral band names/order must match the model; provide feature_names for unnamed rasters")


def _normalize_context(ctx, metadata):
    if ctx is None:
        return None
    result = ctx.copy()
    finite = np.isfinite(result)
    if metadata.get("normalize", True):
        mins = metadata.get("context_mins")
        maxs = metadata.get("context_maxs")
        if mins is None or maxs is None or len(mins) != result.shape[1] or len(maxs) != result.shape[1]:
            raise ValueError("Missing or inconsistent context normalization metadata")
        enabled = metadata.get("context_normalized_bands", [True] * result.shape[1])
        for col, (lo, hi, use) in enumerate(zip(mins, maxs, enabled)):
            if use:
                result[:, col] = (result[:, col] - lo) / (hi - lo) if hi > lo else 0
                result[:, col] = np.clip(result[:, col], 0, 1)
    result[~finite] = np.nan
    return result


def _finite_median(values):
    values = values[np.isfinite(values)]
    return np.median(values) if len(values) else np.nan


def _masked_median(values, size):
    if size == 1:
        return values.copy()
    valid = np.isfinite(values)
    if valid.all():
        return ndimage.median_filter(values, size=size, mode="reflect")
    result = ndimage.generic_filter(values, _finite_median, size=size, mode="reflect")
    result[~valid] = np.nan
    return result


def _detect_arrays(pre, post, ctx, pkg, h, w, *, sigma, filter_size, output_mode,
                   k_neighbors, use_pre_for_clustering=None):
    c = pre.shape[1]
    metadata = pkg.get("context_meta", {})
    use_pre = metadata.get("use_pre_for_clustering", True)
    if use_pre_for_clustering is not None and use_pre_for_clustering != use_pre:
        raise ValueError("use_pre_for_clustering differs from the model's training schema")
    valid = np.isfinite(pre).all(axis=1) & np.isfinite(post).all(axis=1)
    if ctx is not None:
        valid &= np.isfinite(ctx).all(axis=1)
    elif not use_pre:
        raise ValueError("Model requires context-only features")
    ctx = _normalize_context(ctx, metadata)
    source_valid = valid.copy()
    scores = np.full((h*w, c), np.nan, dtype=np.float32)
    if valid.any():
        before, after = pre[valid], post[valid]
        transform, params = resolve_algo("dynamic_system", pkg.get("algorithm_config"))
        values = np.asarray(transform(before, after, **params), dtype=np.float32)
        if values.shape != before.shape:
            raise ValueError("Dynamic system must preserve shape (pixels, source_bands)")
        if metadata.get("modeling_target", "delta") == "post":
            values = after
        target_finite = np.isfinite(values).all(axis=1)
        rows = np.flatnonzero(valid)[target_finite]
        values = values[target_finite]
        features = pre[rows]
        if ctx is not None:
            features = np.hstack([features, ctx[rows]]) if use_pre else ctx[rows]
        if len(rows):
            neighbors = _compute_topk(pkg["model"], features, k_neighbors)
            output = np.full(values.shape, np.nan, dtype=np.float32)
            if "score_model" in pkg:
                for column in range(neighbors.shape[1]):
                    labels = neighbors[:, column]
                    mask = labels >= 0
                    if mask.any():
                        scored = np.asarray(pkg["score_model"].score(values[mask], labels[mask], sigma=sigma), dtype=np.float32)
                        if scored.shape != values[mask].shape:
                            raise ValueError("score() must return shape (pixels, bands)")
                        scored[~np.isfinite(scored)] = np.nan
                        output[mask] = np.fmin(output[mask], scored)
            else:
                models = pkg["gmm_list"]
                scorer = get_scorer(pkg.get("modeler_name", "gmm_bic_model_1d"))
                for category in np.unique(neighbors):
                    if category < 0 or category >= len(models) or models[category] is None:
                        continue
                    mask = (neighbors == category).any(axis=1)
                    if scorer is not None:
                        scored = np.asarray(scorer(models[category], values[mask], sigma=sigma), dtype=np.float32)
                    else:
                        name = pkg.get("modeler_name", "gmm_bic_model_1d")
                        if name not in ("gmm_bic_model_1d", "gmm_bic_model_post"):
                            raise ValueError(f"Modeler '{name}' needs a registered scorer or fit/score protocol")
                        _, scored = _worker_calc_score(category, values[mask], models[category], sigma)
                    if scored.shape != values[mask].shape:
                        raise ValueError("Scorer must return shape (pixels, bands)")
                    scored[~np.isfinite(scored)] = np.nan
                    output[mask] = np.fmin(output[mask], scored)
            scores[rows] = output
    score_valid = np.isfinite(scores).any(axis=1).reshape(h, w)
    raw = scores.reshape(h, w, c)
    band_scores = np.stack([_masked_median(raw[:, :, b], filter_size) for b in range(c)], axis=2)
    pooled = np.fmax.reduce(band_scores, axis=2)
    anomaly = _masked_median(pooled, filter_size)
    anomaly[~score_valid] = np.nan
    result = {"anomaly_map": anomaly, "valid_mask": source_valid.reshape(h, w),
              "score_valid_mask": score_valid}
    if output_mode == "full":
        result["band_scores"] = np.concatenate([band_scores, anomaly[:, :, None]], axis=2)
    return result


def _read_context(context, pkg, info, window=None, cache=None):
    metadata = pkg.get("context_meta", {})
    expected = metadata if metadata.get("context_names") else {"context_names": [], "context_band_counts": []}
    return read_context_stack(context, info[0], info[1], reference=info, expected=expected,
                              window=window, ds_cache=cache)[0]


def detect(pre_path, post_path, bundle, sigma=2.0, filter_size=5, n_jobs=-1,
           output_mode="max", verbose=True, context=None, k_neighbors=1,
           use_pre_for_clustering=None, feature_names=None):
    """Return scores plus separate data-valid and score-valid masks.

    Missing models/classes remain NaN. Median smoothing excludes missing neighbors.
    n_jobs is retained for source compatibility; this shared kernel runs serially.
    """
    _check_options(sigma, filter_size, output_mode, k_neighbors)
    pkg = normalize_bundle(bundle)
    info = validate_pair(pre_path, post_path)
    _check_schema(pkg, pre_path, info, feature_names)
    h, w, _, transform, projection = info
    pre = flat_block(pre_path, 0, 0, w, h)
    post = flat_block(post_path, 0, 0, w, h)
    ctx = _read_context(context, pkg, info)
    result = _detect_arrays(pre, post, ctx, pkg, h, w, sigma=sigma, filter_size=filter_size,
                            output_mode=output_mode, k_neighbors=k_neighbors,
                            use_pre_for_clustering=use_pre_for_clustering)
    result.update(geo_transform=transform, projection=projection)
    if verbose:
        print(f"[TLG Detect] scored {result['score_valid_mask'].sum()} / {h*w} pixels")
    return result


def detect_to_file(pre_path, post_path, bundle, path, *, block_size=512, sigma=2.0,
                   filter_size=5, output_mode="max", context=None, k_neighbors=1,
                   feature_names=None, verbose=True, mask_path=None):
    """Stream tiled scores and a two-band valid/score-valid mask file.

    Two median passes require halo = 2 * floor(filter_size/2). Output tile
    interiors match detect(), including dataset edges. Registered transforms
    must declare __window_safe__=True (pixelwise, no batch-dependent statistics).
    Memory is bounded by a tile plus halo and the already loaded model.
    """
    _check_options(sigma, filter_size, output_mode, k_neighbors)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    pkg = normalize_bundle(bundle)
    dynamic, _ = resolve_algo("dynamic_system", pkg.get("algorithm_config"))
    if not getattr(dynamic, "__window_safe__", False):
        raise ValueError("Window detection requires a dynamic system declaring __window_safe__=True")
    info = validate_pair(pre_path, post_path)
    _check_schema(pkg, pre_path, info, feature_names)
    h, w, c, transform, projection = info
    output = Path(path)
    masks = Path(mask_path) if mask_path else output.with_name(output.stem + ".masks.tif")
    if output.resolve() == masks.resolve():
        raise ValueError("Score output and mask output must have different paths")
    sources = [Path(pre_path), Path(post_path)] + [
        Path(value) for value in _named_context(context).values()
        if isinstance(value, (str, os.PathLike))]
    for source in sources:
        if source.resolve() in (output.resolve(), masks.resolve()):
            raise ValueError("Output cannot overwrite an input raster")
    token = uuid.uuid4().hex
    temporary = output.with_name(output.name + "." + token + ".tmp.tif")
    temporary_mask = masks.with_name(masks.name + "." + token + ".tmp.tif")
    output.parent.mkdir(parents=True, exist_ok=True)
    masks.parent.mkdir(parents=True, exist_ok=True)
    ds = mask_ds = dataset = None
    cache = {}
    count_valid = count_scored = 0
    try:
        ds = gdal.GetDriverByName("GTiff").Create(str(temporary), w, h, c + 1 if output_mode == "full" else 1,
                                                 gdal.GDT_Float32, options=["TILED=YES", "COMPRESS=DEFLATE"])
        mask_ds = gdal.GetDriverByName("GTiff").Create(str(temporary_mask), w, h, 2, gdal.GDT_Byte,
                                                      options=["TILED=YES", "COMPRESS=DEFLATE"])
        for dataset in (ds, mask_ds):
            dataset.SetGeoTransform(transform)
            dataset.SetProjection(projection)
        # Python retains a loop variable after the loop. Releasing only ds and
        # mask_ds leaves this final reference open, preventing rename on Windows.
        dataset = None
        for band in range(ds.RasterCount):
            ds.GetRasterBand(band + 1).SetNoDataValue(float("nan"))
        mask_ds.GetRasterBand(1).SetDescription("valid_mask")
        mask_ds.GetRasterBand(2).SetDescription("score_valid_mask")
        halo = filter_size - 1
        for x, y, bw, bh in iter_windows(h, w, block_size):
            left, top = max(0, x - halo), max(0, y - halo)
            right, bottom = min(w, x + bw + halo), min(h, y + bh + halo)
            win = (left, top, right - left, bottom - top)
            pre = flat_block(pre_path, *win, ds_cache=cache)
            post = flat_block(post_path, *win, ds_cache=cache)
            ctx = _read_context(context, pkg, info, win, cache)
            result = _detect_arrays(pre, post, ctx, pkg, win[3], win[2], sigma=sigma,
                                    filter_size=filter_size, output_mode=output_mode,
                                    k_neighbors=k_neighbors)
            ys, xs = slice(y-top, y-top+bh), slice(x-left, x-left+bw)
            scores = result.get("band_scores", result["anomaly_map"][:, :, None])[ys, xs]
            for band in range(scores.shape[2]):
                ds.GetRasterBand(band + 1).WriteArray(scores[:, :, band], x, y)
            for band, key in enumerate(("valid_mask", "score_valid_mask"), 1):
                mask_ds.GetRasterBand(band).WriteArray(result[key][ys, xs].astype(np.uint8), x, y)
            count_valid += int(result["valid_mask"][ys, xs].sum())
            count_scored += int(result["score_valid_mask"][ys, xs].sum())
        ds.FlushCache()
        mask_ds.FlushCache()
        ds = mask_ds = dataset = None
        os.replace(temporary_mask, masks)
        os.replace(temporary, output)
    finally:
        ds = mask_ds = dataset = None
        cache.clear()
        for temp in (temporary, temporary_mask):
            # Preserve the original inference/publication exception if cleanup
            # itself fails (for example, a third party briefly holds a handle).
            with suppress(OSError):
                if temp.exists():
                    temp.unlink()
    if verbose:
        print(f"[TLG Detect] streamed {count_scored} / {h*w} scored pixels to {output}")
    return {"path": str(output), "mask_path": str(masks), "valid_pixels": count_valid,
            "scored_pixels": count_scored, "geo_transform": transform, "projection": projection}
