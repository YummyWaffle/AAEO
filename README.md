# AAEO: Attention Allocation in Earth Observation

AAEO provides an indicator of the discrepancy between **observed land-surface processes** and
**geographically expected land-surface processes**, supporting decisions about which observations
merit attention.

The library provides `tlg_api` for learning a reference representation and computing the indicator,
and `tlg_core` for shared data contracts. It operates on prepared, aligned Earth observation imagery.

## Concept and interpretation

The purpose of AAEO is attention allocation in Earth observation. The indicator expresses how far
an observation departs from the expected surface process in its geographical context. A higher score
indicates greater discrepancy relative to the chosen reference model and scoring configuration,
providing a basis for deciding where to direct further examination.

In the current TLG implementation, paired observations represent the observed surface process.
The default transform computes `post - pre`; clustering uses the earlier observation and, when
provided, geographical context features. Per-cluster statistical models learned from the training
pair provide an operational representation of expected processes. The meaning of this expectation
therefore depends on the training observations, features, context, and model configuration.

The indicator supports attention decisions; it does not itself establish the cause or significance
of an observation. It is not a calibrated probability or a universal threshold for attention.

Data acquisition is outside this library's scope. Experimental datasets will be released separately.
This repository contains no `tlg_data` package, download tools, real experimental imagery, or pretrained models.

## Installation

The source is distributed through GitHub. After downloading or cloning the repository, install from
its root directory into your Python environment:

```sh
python -m pip install .
```

The distribution name is `aaeo-tlg`. The Python import names are `tlg_api` and `tlg_core`:

```python
from tlg_api import train, detect, detect_to_file, save_bundle, load_bundle
from tlg_core import GridSpec, BandSpec
```

Requirements: Python 3.10+, NumPy, SciPy, scikit-learn, joblib, and GDAL.
The GDAL Python bindings must match the installed GDAL library. An existing conda geospatial
environment can provide these dependencies. If all dependencies are already available, use
`python -m pip install --no-deps .` to install without resolving them again.

Avoid installing another project that provides the same `tlg_api` package in this environment.
Use `print(tlg_api.__file__)` to check which installation Python imports.

FAISS is optional, but **the default `train()` configuration uses FAISS**. To use that configuration,
install a compatible FAISS build, or install the optional dependency with:

```sh
python -m pip install ".[faiss]"
```

Without FAISS, select MiniBatchKMeans explicitly as shown below. Plotting with `anomaly_preview()`
requires the visualization dependencies:

```sh
python -m pip install ".[viz]"
```

## Quick start

```python
from tlg_api import train, save_bundle, load_bundle, detect, save_detection

bundle = train(
    "pre.tif", "post.tif",
    config={
        "extractor": {
            "mini_batch_k_means_extractor": {"component_number": 20}
        }
    },
    max_workers=2,
)
save_bundle(bundle, "model.pkg")

model = load_bundle("model.pkg")
result = detect("pre.tif", "target.tif", model, sigma=2.0, filter_size=5)
save_detection(result, "aaeo_score.tif")
```

`post.tif` is the later image used for training. `target.tif` is the later image to evaluate;
it may be the same file or another image meeting the same feature requirements.

The default GMM modeler requires at least 1,000 training samples per cluster. Small images or too
many clusters can leave some categories without a fitted model. For a small working example,
use the synthetic workflow below.

## Synthetic example

After installation, run:

```sh
python -B examples/minimal_example.py
```

The example uses GDAL to generate 64 x 64 images with three bands, then trains a model, saves and
reloads it, and computes the indicator over the full image and by window. It checks that a region
with an injected departure from the learned process has a higher score than the reference background
and that both scoring paths agree.

Outputs are written to `artifacts/minimal/`: input images, a model bundle, an AAEO indicator raster
(`anomaly.tif`), a two-band validity mask raster, and `summary.json`. The example uses MiniBatchKMeans.

With FAISS installed, run:

```sh
python -B examples/minimal_example.py --extractor faiss --output artifacts/minimal-faiss
```

## Input and output contracts

- Input images must have the same CRS, transform, dimensions, band count, and band meanings.
  Identically named bands from different sensors are not automatically interchangeable.
- Band names and order must match the trained model. Supply auxiliary rasters explicitly, for example
  `context={"dem": "dem.tif"}`. These rasters must share the image grid.
- Declare categorical features with `context_types={"lulc": "categorical"}` during training.
  Scoring reuses the saved feature schema and normalization metadata.
- Valid zeros are preserved. Declared NoData, masked pixels, and non-finite values are excluded from scoring.
- `valid_mask` describes valid inputs; `score_valid_mask` identifies pixels with at least one scoreable band.
  Missing models remain unscoreable rather than receiving zero discrepancy scores.
- Default GMM scores quantify discrepancy from the learned reference distributions used to represent
  expected surface processes. Lower `sigma` values give higher scores for the same deviation.
- `detect()` returns arrays in memory. For large rasters, `detect_to_file()` writes score GeoTIFFs
  and a separate `.masks.tif` file by window.
- `train(block_size=..., max_training_samples=...)` performs bounded sampling, not incremental fitting.
- Model `.pkg` files use joblib and must come from trusted sources. Compatibility across dependency
  versions is not guaranteed.

See [API details](docs/API_DETAILS.md) for the interface documentation and
[validation results](docs/VALIDATION.md) for the tested scope.

## Tests and distribution checks

With runtime dependencies and pytest installed, run the existing tests:

```sh
python -B -m pytest tests --basetemp=.work/pytest-temp -q
```

Local test tools, installation targets, caches, and outputs are kept in `.work/` or `artifacts/`.
These directories are excluded from Git and from the library wheel.

The distribution verification script builds a wheel offline, installs it into `.work/installed`,
and runs the example from an isolated Python process. It uses existing dependencies without
installing the project into the active environment or downloading packages:

```sh
python -B scripts/validate_distribution.py
```

This checks MiniBatchKMeans; add `--faiss` to also check FAISS.

## Reproducibility

Experimental data will be published separately. See the [reproducibility guide](reproducibility/README.md)
for the information needed to reproduce an experiment. Dataset links and real-data reproduction
results are not yet included. The synthetic example verifies execution; it does not establish the usefulness of the indicator for allocating attention in real observations.

## License

A license has not yet been selected. No open-source license grant should be inferred from this snapshot.
Licensing terms will be determined by the project owner.
