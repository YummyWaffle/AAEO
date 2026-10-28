"""Offline correctness, extension, persistence and streaming contracts."""
from pathlib import Path
import numpy as np
import pytest
from osgeo import gdal, osr
from tlg_api import (train, detect, detect_to_file, save_bundle, load_bundle,
                     save_detection, register_dynamic_system, register_extractor)
from tlg_api._io import read_image_full, read_image_pair_chunked, write_geotiff
from tlg_api._detect import _compute_topk
from tlg_api.algorithms._extractors._lulc_stratified import LulcStratifiedKMeans

TRANSFORM = (100, 30, 0, 900, 0, -30)


class OneCluster:
    n_clusters = 1
    __window_safe__ = True

    def predict(self, values):
        return np.zeros(len(values), dtype=np.int32)

    def predict_neighbors(self, values, k):
        return self.predict(values)[:, None]


def one_cluster(values):
    return np.zeros(len(values), dtype=np.int32), OneCluster()


def ratio(pre, post):
    return post / pre


ratio.__window_safe__ = True
register_extractor("test_one_cluster", one_cluster)
register_dynamic_system("test_ratio", ratio)


def config(dynamic="difference_dynamic"):
    return {"extractor": {"test_one_cluster": {}},
            "modeler": {"robust_deviation": {"min_samples": 2}},
            "dynamic_system": {dynamic: {}}}


def raster(tmp_path, name, data, *, transform=TRANSFORM, nodata=np.nan, names=None):
    path = tmp_path / name
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(32649)
    write_geotiff(path, np.asarray(data, dtype=np.float32), transform,
                  spatial.ExportToWkt(), descriptions=names, nodata=nodata)
    return path


def legacy_style_bundle(bands=1):
    return {"schema_version": 2, "model": OneCluster(),
            "gmm_list": [[(np.array([0]), np.array([1]), np.array([1])) for _ in range(bands)]],
            "algorithm_config": {"dynamic_system": {"difference_dynamic": {}}},
            "modeler_name": "gmm_bic_model_1d",
            "feature_schema": {"source_bands": bands}}


def test_single_band_nodata_and_infinity(tmp_path):
    data = np.array([[0, 1], [-9999, np.inf]], dtype=np.float32)
    path = raster(tmp_path, "single.tif", data, nodata=-9999)
    flat, h, w, c, _, _ = read_image_full(path)
    assert (h, w, c) == (2, 2, 1)
    assert flat.shape == (4, 1)
    assert flat[0, 0] == 0
    assert np.isnan(flat[2:]).all()
    chunks = list(read_image_pair_chunked(path, path, block_size=1))
    assert len(chunks) == 4 and chunks[0][0].shape == (1, 1)


@pytest.mark.parametrize("field", ["grid", "bands"])
def test_grid_and_band_mismatch_rejected(tmp_path, field):
    pre = raster(tmp_path, "pre.tif", np.ones((4, 4)))
    shape = (4, 4, 2) if field == "bands" else (4, 4)
    transform = (115, 30, 0, 900, 0, -30) if field == "grid" else TRANSFORM
    post = raster(tmp_path, "post.tif", np.ones(shape), transform=transform)
    with pytest.raises(ValueError):
        train(pre, post, config=config(), verbose=False)
    with pytest.raises(ValueError):
        detect(pre, post, legacy_style_bundle(), verbose=False)


def test_max_full_masks_and_output_nodata(tmp_path):
    data = np.ones((7, 8))
    data[0, 0] = np.nan
    pre = raster(tmp_path, "pre.tif", data)
    post = raster(tmp_path, "post.tif", data + 3)
    bundle = legacy_style_bundle()
    result = detect(pre, post, bundle, filter_size=3, verbose=False)
    full = detect(pre, post, bundle, filter_size=3, output_mode="full", verbose=False)
    np.testing.assert_allclose(result["anomaly_map"], full["band_scores"][:, :, -1], equal_nan=True)
    assert not result["valid_mask"][0, 0]
    assert not result["score_valid_mask"][0, 0]
    assert np.isnan(result["anomaly_map"][0, 0])
    path = tmp_path / "out.tif"
    save_detection(result, path)
    ds = gdal.Open(str(path))
    assert np.isnan(ds.GetRasterBand(1).GetNoDataValue())
    assert np.isnan(ds.ReadAsArray()[0, 0])


def test_missing_model_is_unscorable(tmp_path):
    pre = raster(tmp_path, "pre.tif", np.ones((3, 3)))
    bundle = legacy_style_bundle()
    bundle["gmm_list"] = [None]
    result = detect(pre, pre, bundle, filter_size=1, verbose=False)
    assert result["valid_mask"].all()
    assert not result["score_valid_mask"].any()
    assert np.isnan(result["anomaly_map"]).all()


def test_context_order_reordered_and_schema_checked(tmp_path):
    values = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    pre = raster(tmp_path, "pre.tif", values)
    post = raster(tmp_path, "post.tif", values + 2)
    a, b = values.reshape(-1, 1), (values * 2).reshape(-1, 1)
    bundle = train(pre, post, config=config(), context={"dem": a, "slope": b}, verbose=False)
    first = detect(pre, post, bundle, context={"dem": a, "slope": b}, filter_size=1, verbose=False)
    swapped = detect(pre, post, bundle, context={"slope": b, "dem": a}, filter_size=1, verbose=False)
    np.testing.assert_allclose(first["anomaly_map"], swapped["anomaly_map"])
    with pytest.raises(ValueError, match="names mismatch"):
        detect(pre, post, bundle, context={"wrong": a, "slope": b}, verbose=False)
    shifted = raster(tmp_path, "context.tif", values, transform=(101, 30, 0, 900, 0, -30))
    with pytest.raises(ValueError, match="transform"):
        train(pre, post, config=config(), context={"dem": shifted}, verbose=False)


def test_dynamic_saved_and_non_gmm_roundtrip(tmp_path):
    values = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    pre = raster(tmp_path, "pre.tif", values)
    post = raster(tmp_path, "post.tif", values * 2)
    bundle = train(pre, post, config=config("test_ratio"), verbose=False)
    assert "score_model" in bundle and bundle["schema_version"] == 2
    path = tmp_path / "ratio.pkg"
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    result = detect(pre, post, loaded, filter_size=1, verbose=False)
    np.testing.assert_allclose(result["anomaly_map"], 0, atol=1e-6)


@pytest.mark.parametrize("mode", ["max", "full"])
@pytest.mark.parametrize("size", [1, 3, 5])
def test_streaming_matches_full_with_halo(tmp_path, mode, size):
    rng = np.random.default_rng(11)
    values = rng.uniform(0.1, 1, (15, 19, 2)).astype(np.float32)
    after = values + rng.uniform(0, 10, values.shape).astype(np.float32)
    values[4:6, 5:7] = np.nan
    after[9, 12] = np.nan
    pre = raster(tmp_path, "pre.tif", values)
    post = raster(tmp_path, "post.tif", after)
    bundle = legacy_style_bundle(2)
    expected = detect(pre, post, bundle, filter_size=size, output_mode=mode, verbose=False)
    output = tmp_path / f"stream_{mode}_{size}.tif"
    summary = detect_to_file(pre, post, bundle, output, block_size=4, filter_size=size,
                             output_mode=mode, verbose=False)
    actual, h, w, c, _, _ = read_image_full(output)
    expected_scores = expected.get("band_scores", expected["anomaly_map"][:, :, None])
    np.testing.assert_allclose(actual.reshape(h, w, c), expected_scores, atol=1e-7, equal_nan=True)
    masks = gdal.Open(summary["mask_path"]).ReadAsArray()
    np.testing.assert_array_equal(masks[0], expected["valid_mask"])
    np.testing.assert_array_equal(masks[1], expected["score_valid_mask"])


def test_bounded_training_with_context_and_seed(tmp_path):
    values = np.arange(1, 401, dtype=np.float32).reshape(20, 20)
    pre = raster(tmp_path, "pre.tif", values)
    post = raster(tmp_path, "post.tif", values + values / 10)
    ctx = {"dem": values.reshape(-1, 1)}
    a = train(pre, post, config=config(), context=ctx, block_size=4,
              max_training_samples=40, random_state=13, verbose=False)
    b = train(pre, post, config=config(), context=ctx, block_size=4,
              max_training_samples=40, random_state=13, verbose=False)
    assert a["training_meta"]["valid_pixels"] == 400
    assert a["training_meta"]["training_samples"] == 40
    np.testing.assert_equal(a["score_model"].parameters, b["score_model"].parameters)


def test_lulc_neighbors_never_cross_class():
    class LocalModel:
        n_clusters = 1
        cluster_centers_ = np.array([[0, 0]], dtype=np.float32)
        def predict_neighbors(self, values, k):
            return np.zeros((len(values), 1), dtype=np.int32)
    model = LulcStratifiedKMeans({1: LocalModel(), 2: LocalModel()}, {1: 0, 2: 1},
                               2, -1, [1, 2])
    actual = _compute_topk(model, np.array([[0, 1], [0, 2], [0, 3]], dtype=np.float32), 2)
    np.testing.assert_array_equal(actual, [[0, -1], [1, -1], [-1, -1]])


@pytest.mark.parametrize("bad", [
    {"extractor": {}}, {"oops": {}},
    {"extractor": {"test_one_cluster": {"typo": 2}}},
    {"extractor": {"test_one_cluster": {}, "other": {}}},
    {"extractor": {"test_one_cluster": []}},
])
def test_bad_config_fails_before_io(bad):
    with pytest.raises((ValueError, TypeError)):
        train("does-not-exist.tif", "does-not-exist.tif", config=bad, verbose=False)


def test_legacy_bundle_adapter(tmp_path):
    import joblib
    old = legacy_style_bundle()
    old.pop("schema_version")
    old.pop("algorithm_config")
    path = tmp_path / "legacy.pkg"
    joblib.dump(old, path)
    with pytest.warns(UserWarning, match="Legacy bundle"):
        restored = load_bundle(path)
    assert restored["legacy_assumptions"] == ["difference_dynamic"]


def test_invalid_detection_options(tmp_path):
    with pytest.raises(ValueError, match="odd"):
        detect("missing.tif", "missing.tif", {}, filter_size=2)


def test_named_spectral_schema_enforced(tmp_path):
    data = np.ones((4, 4, 2))
    pre = raster(tmp_path, "pre.tif", data, names=["red", "nir"])
    bundle = train(pre, pre, config=config(), verbose=False)
    reverse = raster(tmp_path, "reverse.tif", data, names=["nir", "red"])
    with pytest.raises(ValueError, match="names/order"):
        detect(reverse, reverse, bundle, verbose=False)


def test_streaming_context_and_nested_output(tmp_path):
    values = np.arange(1, 145, dtype=np.float32).reshape(12, 12)
    pre = raster(tmp_path, "pre.tif", values)
    post = raster(tmp_path, "post.tif", values + np.sin(values))
    ctx = raster(tmp_path, "dem.tif", values * 10)
    bundle = train(pre, post, config=config(), context={"dem": ctx}, verbose=False)
    expected = detect(pre, post, bundle, context={"dem": ctx}, filter_size=3, verbose=False)
    path = tmp_path / "nested" / "stream.tif"
    detect_to_file(pre, post, bundle, path, context={"dem": ctx}, block_size=3,
                   filter_size=3, verbose=False)
    actual, h, w, _, _, _ = read_image_full(path)
    np.testing.assert_allclose(actual.reshape(h, w), expected["anomaly_map"], equal_nan=True)
    with pytest.raises(ValueError, match="overwrite"):
        detect_to_file(pre, post, bundle, ctx, context={"dem": ctx}, verbose=False)


def test_import_without_visualization_dependencies():
    """A fresh interpreter must support ML without importing the viz extra."""
    import os
    import subprocess
    import sys
    source = """
import importlib.abc
import sys
class BlockVisualization(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'PIL'}:
            raise ModuleNotFoundError('Visualization dependency intentionally blocked', name=fullname)
sys.meta_path.insert(0, BlockVisualization())
import tlg_api
assert callable(tlg_api.train)
assert callable(tlg_api.detect)
assert 'tlg_api._viz' not in sys.modules
assert 'matplotlib' not in sys.modules
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-B", "-c", source],
                               cwd=Path(__file__).resolve().parents[1],
                               env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("bad_config", [None, {}, {"dynamic_system": {}},
                                       {"dynamic_system": {"difference_dynamic": []}}])
def test_version_two_requires_explicit_dynamic_schema(bad_config):
    from tlg_api._bundle import normalize_bundle
    bundle = legacy_style_bundle()
    bundle["algorithm_config"] = bad_config
    with pytest.raises(ValueError, match="dynamic"):
        normalize_bundle(bundle)


def test_legacy_explicit_transform_is_preserved_without_false_assumption():
    from tlg_api._bundle import normalize_bundle
    bundle = legacy_style_bundle()
    bundle.pop("schema_version")
    bundle["algorithm_config"] = {"dynamic_system": {"test_ratio": {}}}
    with pytest.warns(UserWarning, match="supplied dynamic"):
        adapted = normalize_bundle(bundle)
    assert adapted["algorithm_config"] == bundle["algorithm_config"]
    assert "legacy_assumptions" not in adapted
    assert "schema_version" not in bundle
