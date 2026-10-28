# AAEO API reference

AAEO (Attention Allocation in Earth Observation) uses an indicator of discrepancy between
observed and geographically expected land-surface processes to support attention decisions.
The reference expectation is learned from the selected training observations and features, with
optional geographical context. Its interpretation depends on those inputs and the model configuration.
This document describes the implemented computational interface.

## Learning the reference and computing the indicator

```python
from tlg_api import train, save_bundle, detect, detect_to_file

bundle = train(
    "pre.tif", "post.tif",
    config={
        "extractor": {"mini_batch_k_means_extractor": {"component_number": 20}},
        "modeler": {"robust_deviation": {"min_samples": 20}},
    },
    context={"dem": "dem.tif", "lulc": "lulc.tif"},
    context_types={"lulc": "categorical"},
    block_size=512,
    max_training_samples=250_000,
    random_state=42,
)
save_bundle(bundle, "model.pkg")
result = detect("pre.tif", "post.tif", bundle,
                context={"lulc": "lulc.tif", "dem": "dem.tif"})
summary = detect_to_file("pre.tif", "post.tif", bundle, "scores.tif",
                         context={"dem": "dem.tif", "lulc": "lulc.tif"},
                         block_size=512)
```

The raster pair must have equal CRS, transform, dimensions and band count.
Band descriptions are validated when present; feature_names can declare the
order for unnamed data. Named context inputs are reordered to the saved schema
and each raster is checked against the reference grid. Categorical context is
never normalized or clipped. Pass context_types explicitly for categorical data.

block_size enables bounded, uniform random-priority sampling, not incremental
fitting. The implicit cap in block mode is 250,000 valid pixels. Context rasters
are also read by window. Memory depends on this cap, tile size, feature count and
the selected algorithm. Full mode remains available without a cap. Training
metadata records the sample cap, seed, valid count and selected sample count.

## Scores and missing data

Declared NoData, GDAL masks and non-finite values become NaN. Zero is valid.
valid_mask describes valid source/context pixels; score_valid_mask describes
pixels for which at least one band can be scored. Unknown classes and absent
models remain NaN. max and full modes share the same output computation.
full returns an (H, W, C+1) array, with the `anomaly_map` result last.

Median filtering excludes missing neighbors and preserves each missing center.
The implementation uses a slower NaN-aware filter for tiles containing missing
scores. n_jobs remains accepted for compatibility; inference is currently serial.
Two smoothing passes require a halo of filter_size - 1 pixels. detect_to_file()
writes scores and a separate two-band .masks.tif without allocating full rasters.
The output pair is written via temporary files; each rename is atomic, but the
two-file publication is not a cross-file transaction.

## Algorithm extensions

Registry configuration rejects unknown stages, multiple algorithms per stage,
unknown parameters and arguments managed by the pipeline. Custom parameters
must be explicit in the function signature (or listed in __parameter_names__).
Custom algorithms must be registered before loading/using the model package.

Dynamic functions receive (pre, post, **params), return the same pixel/band
shape, and have their name/parameters stored in the bundle. Pixelwise transforms
can declare __window_safe__ = True to enable streaming inference. Transforms
that depend on whole-image batch statistics must use full-image scoring or implement
a separate fitted transform. The built-in difference transform is window safe.

Extractors receive a feature matrix and return labels plus a fitted model.
predict_neighbors(X, k) owns nearest-neighbor semantics and returns -1 for
unknown classes. LULC-stratified extraction only returns neighbors in the same
class. Generic custom extractors without this method support k_neighbors=1.

Statistical model factories expose __tlg_fit_score__ = True and return an object
with fit(values, labels, components) and score(values, labels, sigma=2.0).
Register the factory using register_modeler(name, factory). The built-in
robust_deviation model is a working non-GMM example. The interfaces module
provides structural protocols; subclassing is optional.

Legacy modeling functions remain supported. Register a companion
scorer(models_for_component, values, sigma=...) with
register_modeler(name, training_function, scorer=scoring_function). Both score
interfaces return (pixels, source_bands), using NaN for unavailable scores.
Built-in GMMs have an adapter. A custom modeler without a scorer is rejected at
inference instead of being silently interpreted as GMM.

## Model persistence and compatibility

New bundles have schema_version=2, algorithm_config, feature_schema,
context_meta and training_meta. Custom fit/score objects use score_model; legacy
GMMs retain model and gmm_list. Saves use a sibling temporary file and atomic
replacement. Joblib packages are trusted Python objects, not a safe exchange
format for untrusted inputs.

Unversioned bundles are accepted with an explicit warning that the missing
dynamic configuration is assumed to be difference_dynamic. Historical custom
transforms cannot be recovered automatically; supply their configuration in a
reviewed migration. Unsupported future schema versions fail explicitly.

The old post-target modeler remains compatible, with post values passed
consistently even when a custom dynamic function is configured. Custom modeler
factories can declare __modeling_target__ = "post".

## Verification and limitations

tests/test_api_reconstruction.py uses offline synthetic rasters to check I/O,
schema mismatches, masks, custom dynamics, non-GMM persistence, LULC neighbors,
bounded sampling and exact tile/full equivalence for filter sizes 1, 3 and 5.
These tests do not establish production-scale throughput or online integration.
The full API still uses GDAL. GMM and MiniBatch KMeans require scikit-learn;
FAISS is imported only when that extractor is used.
