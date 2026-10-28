# Distribution scope

The distribution provides `tlg_api` and `tlg_core`, installation metadata, tests, a synthetic example,
and English documentation for AAEO: Attention Allocation in Earth Observation.

It excludes `tlg_data`, download tools, internal experiment projects, real datasets, and pretrained
models. Experimental data will be published separately.

## Source verification

All 23 Python files in `tlg_api` and `tlg_core` are byte-identical to the source project
`tlg_franky_reconstruct` 0.3.0. The two distributed test files are also unchanged.
The source manifest records their SHA-256 checksums. The API reference is maintained separately
as documentation.

## Local outputs

`.work/`, `artifacts/`, `build/`, `dist/`, and `*.egg-info/` contain local verification outputs and are
excluded by `.gitignore`. Only the non-ignored source, tests, examples, configuration, and documentation
belong in the GitHub repository.

All preparation outputs are confined to `aaeo_distribute_v1`. The original project, reconstructed
checkout, and existing Python environment are untouched.
