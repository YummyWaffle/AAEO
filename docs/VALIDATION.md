# Distribution validation

Validation date: 2026-10-28 (Asia/Shanghai). Numerical algorithms and default parameters were unchanged.

## Results

| Check | Result |
| --- | --- |
| Python source in both packages | All 23 files are byte-identical to source; wheel hashes match the source manifest |
| Tests | Both test files are byte-identical to source |
| Existing algorithm tests | 39 passed, 0 failed, 0 skipped |
| Local wheel build and installation | Passed; installed only into `.work/installed` |
| Wheel contents | Only `tlg_api`, `tlg_core`, and installation metadata; no `tlg_data` |
| External imports | Isolated processes started in `.work/external-cwd`; both packages imported from `.work/installed` |
| Independence from the acquisition library | Both examples completed with `tlg_data` imports explicitly blocked |
| Original project preservation | Size and modification-time inventories matched for 100,610 files outside the distribution directory |

The original-project inventory compares file sizes and nanosecond modification times; it is not a
content-hash audit of the entire original dataset. The 23 package files and 2 test files
were additionally checked against their original SHA-256 hashes. The API documentation has its own
distribution checksum. Directory links are not traversed recursively.

## Synthetic end-to-end checks

Inputs were fixed-seed, 64 x 64 images with three bands. Training used two clusters and one GMM
component per cluster per band, with 16-pixel windows and a 4,096-sample cap. Indicator computation used `sigma=2`,
a 3 x 3 median filter, and 16-pixel windows.

| Extractor | Median score in the injected-departure region | Median background score | Valid / scoreable pixels | Full-image and windowed agreement |
| --- | --- | --- | --- | --- |
| MiniBatchKMeans | 0.97757548 | 0.0 | 4096 / 4096 | Passed, absolute tolerance 1e-7 |
| FAISS KMeans | 0.97757548 | 0.0 | 4096 / 4096 | Passed, absolute tolerance 1e-7 |

Both paths included model saving, reloading, in-memory indicator computation, and windowed GeoTIFF output.
The examples selected extractors and parameters explicitly. They do not validate every default
parameter combination or every built-in algorithm.

## Environment

| Component | Version |
| --- | --- |
| Operating system | Windows 11, build 26200 |
| Python | 3.12.13, conda-forge |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| GDAL | 3.12.3 |
| FAISS | 1.10.0, runtime version from the existing conda environment |
| pytest | 8.4.2, local test-tool copy |
| setuptools / wheel | 82.0.1 / 0.47.0 |

The existing Python environment was reused without installing, upgrading, or removing its packages.
pytest was copied from the reconstructed project's existing test tools into `.work/testdeps` for
local verification. It is not a runtime dependency of the distributed library.

## Local evidence

These files are verification artifacts and are excluded from Git:

- `artifacts/validation/junit.xml`: results for the 39 existing tests.
- `artifacts/validation/distribution_validation.json`: package contents, import locations, environment, and example results.
- `artifacts/validation/build.log`, `install.log`, and `isolated-*.log`: build, installation, and isolated execution logs.
- `artifacts/validation/source_verification.json`: source checksums and original-project inventory comparison.
- `artifacts/installed-minibatch/summary.json` and `artifacts/installed-faiss/summary.json`: synthetic example statistics.
- `dist/aaeo_tlg-0.3.0-py3-none-any.whl`: the locally built wheel; it has not been uploaded.

## Scope and limitations

These checks establish independent packaging, isolated imports, and the tested core behavior using
existing local runtime dependencies. They do not cover online dependency installation in a clean
environment, other operating systems, real-data reproduction, performance benchmarks, or all extension
combinations.

All 23 package source files, including docstrings and plot labels, match the original files exactly.
The wheel and local installation were checked against these hashes. The recorded numerical tests
apply to this same source code.

License terms and experimental dataset release information remain to be supplied by the project owner.
