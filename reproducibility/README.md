# Experimental data and reproducibility

Experimental datasets will be released separately. This library does not include a data acquisition
package, and running the algorithms does not require `tlg_data`.

Dataset locations, experiment configurations, and reference results have not yet been supplied.
This document describes what a reproducible experiment release should include; it is not a report
of a completed real-data experiment.

## Experiment record

| Item | Required information |
| --- | --- |
| Code version | A Git tag or full commit hash, rather than only a moving default branch |
| Data version | Dataset location, release version, file manifest, and checksums |
| Input imagery | The relationship between reference-training pre/post images and observations to be scored, acquisition dates, and sensors |
| Input semantics | Band names and order, units, scale/offset, NoData, CRS, and spatial resolution |
| Preprocessing | Alignment, masking, cloud removal, and associated parameters; preferably provide images ready for direct algorithm input |
| Auxiliary data | Sources, versions, alignment procedures, and feature types for DEM, slope, aspect, land cover, or other context |
| Geographical expectation | The reference observations and geographical context used to represent expected surface processes, and the rationale for their selection |
| Algorithm configuration | Dynamic transform, clustering and modeling parameters, sample cap, random seed, sigma, and filter size |
| Environment | Python, GDAL, NumPy, SciPy, scikit-learn, joblib, and FAISS versions, plus CPU/GPU information |
| Execution | Experiment scripts using relative data paths and complete run commands |
| Expected results | AAEO indicator rasters, masks, statistics, numerical tolerances, and the scope of evaluation |
| Attention evaluation | How observations are prioritized from the indicator, and what evidence is used to assess whether they merit attention |

Once the data files are available, reproduction scripts should call `tlg_api` directly without
referencing internal project paths. If only raw satellite data is released, the preprocessing method
must also be published so that the algorithm inputs can be reproduced.

The repository's `examples/minimal_example.py` is a separate synthetic demonstration. It is not a
substitute for reproducing a real-data experiment.
