"""Synthetic reference learning and AAEO scoring; uses only installed tlg_api and GDAL.

Run after installing this project: python -B examples/minimal_example.py
No sys.path injection: imports must resolve through the active environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from tlg_api import train, detect, detect_to_file, save_bundle, load_bundle


def write_raster(path, values):
    height, width, bands = values.shape
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(32611)
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), width, height, bands, gdal.GDT_Float32)
    dataset.SetGeoTransform((500000, 30, 0, 4100000, 0, -30))
    dataset.SetProjection(spatial.ExportToWkt())
    for index, name in enumerate(("blue", "red", "nir08"), 1):
        band = dataset.GetRasterBand(index)
        band.WriteArray(values[:, :, index - 1])
        band.SetNoDataValue(float("nan"))
        band.SetDescription(name)
    dataset.FlushCache()
    dataset = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "minimal")
    parser.add_argument("--extractor", choices=("minibatch", "faiss"), default="minibatch")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    before = rng.normal(0.2, 0.01, (64, 64, 3)).astype("float32")
    before[:, 32:, :] += 0.4
    after = before + rng.normal(0.01, 0.002, before.shape).astype("float32")
    target = after.copy()
    target[16:24, 16:24, :] += 0.2
    for name, values in (("pre", before), ("post", after), ("target", target)):
        write_raster(output / (name + ".tif"), values)
    algorithm = "mini_batch_k_means_extractor" if args.extractor == "minibatch" else "faiss_kmeans_extractor"
    config = {"extractor": {algorithm: {"component_number": 2, "max_iter": 20}},
              "modeler": {"gmm_bic_model_1d": {"max_components": 1}}}
    bundle = train(output / "pre.tif", output / "post.tif", config=config,
                   max_workers=1, block_size=16, max_training_samples=4096, verbose=False)
    save_bundle(bundle, output / "model.pkg")
    loaded = load_bundle(output / "model.pkg")
    result = detect(output / "pre.tif", output / "target.tif", loaded, filter_size=3, verbose=False)
    streamed = detect_to_file(output / "pre.tif", output / "target.tif", loaded,
                              output / "anomaly.tif", block_size=16, filter_size=3, verbose=False)
    dataset = gdal.Open(str(output / "anomaly.tif"))
    score = dataset.ReadAsArray()
    dataset = None
    np.testing.assert_allclose(score, result["anomaly_map"], atol=1e-7, equal_nan=True)
    event = float(np.nanmedian(score[16:24, 16:24]))
    background = float(np.nanmedian(score[32:48, 8:24]))
    assert event > 0.8 and background < 0.1, (event, background)
    assert result["valid_mask"].all() and result["score_valid_mask"].all()
    summary = {"extractor": args.extractor, "event_median": event,
               "background_median": background, "full_window_equal": True,
               "training": bundle["training_meta"], "detection": streamed}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
