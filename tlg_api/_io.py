"""Raster I/O with explicit grid and missing-data conventions."""
import os
from pathlib import Path
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
DEFAULT_BLOCK_SIZE = 4096
NODATA_SENTINEL = -9999.0


def _open(path, cache=None):
    key = os.fspath(path)
    if cache is not None and key in cache:
        return cache[key]
    ds = gdal.Open(key, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"Cannot open raster: {key}")
    if cache is not None:
        cache[key] = ds
    return ds


def get_image_info(path):
    ds = _open(path)
    return (ds.RasterYSize, ds.RasterXSize, ds.RasterCount,
            ds.GetGeoTransform(), ds.GetProjection())


def band_names(path):
    ds = _open(path)
    names = [ds.GetRasterBand(i + 1).GetDescription() for i in range(ds.RasterCount)]
    return names if all(names) else None


def _same_crs(left, right):
    if left == right:
        return True
    if not left or not right:
        return False
    a, b = osr.SpatialReference(), osr.SpatialReference()
    a.ImportFromWkt(left)
    b.ImportFromWkt(right)
    return bool(a.IsSame(b))


def validate_grid(reference, candidate, *, bands=True, name="raster"):
    if reference[:2] != candidate[:2]:
        raise ValueError(f"{name}: raster dimensions differ from the reference grid")
    if bands and reference[2] != candidate[2]:
        raise ValueError(f"{name}: band count differs from the reference")
    if not np.allclose(reference[3], candidate[3], rtol=0, atol=1e-8):
        raise ValueError(f"{name}: transform differs from the reference grid")
    if not _same_crs(reference[4], candidate[4]):
        raise ValueError(f"{name}: CRS differs from the reference grid")


def validate_pair(pre_path, post_path):
    info = get_image_info(pre_path)
    validate_grid(info, get_image_info(post_path), name="post")
    pre_names, post_names = band_names(pre_path), band_names(post_path)
    if pre_names and post_names and pre_names != post_names:
        raise ValueError("pre/post band names or order differ")
    return info


def read_block(path, x, y, w, h, ds_cache=None):
    ds = _open(path, ds_cache)
    raw = ds.ReadAsArray(int(x), int(y), int(w), int(h))
    if raw is None:
        raise IOError(f"Failed reading block from {path}")
    data = np.asarray(raw, dtype=np.float32)
    if data.ndim == 2:
        data = data[None, :, :]
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            data[i][data[i] == nodata] = np.nan
        if band.GetMaskFlags() != gdal.GMF_ALL_VALID:
            valid = band.GetMaskBand().ReadAsArray(int(x), int(y), int(w), int(h))
            data[i][valid == 0] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data


def flat_block(path, x, y, w, h, ds_cache=None):
    block = read_block(path, x, y, w, h, ds_cache)
    return block.transpose(1, 2, 0).reshape(-1, block.shape[0])


def read_image_full(path):
    h, w, c, transform, projection = get_image_info(path)
    return flat_block(path, 0, 0, w, h), h, w, c, transform, projection


def iter_windows(h, w, block_size):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            yield x, y, min(block_size, w - x), min(block_size, h - y)


def read_image_pair_chunked(pre_path, post_path, block_size=DEFAULT_BLOCK_SIZE):
    h, w, *_ = validate_pair(pre_path, post_path)
    cache = {}
    for x, y, bw, bh in iter_windows(h, w, block_size):
        yield (flat_block(pre_path, x, y, bw, bh, cache),
               flat_block(post_path, x, y, bw, bh, cache))


def _named_context(context):
    if context is None:
        return {}
    if isinstance(context, (list, tuple)):
        named = {}
        for path in context:
            name = Path(path).stem
            if name in named:
                raise ValueError(f"Duplicate context name: {name}; use an explicit dict")
            named[name] = path
        return named
    if not isinstance(context, dict):
        raise TypeError("context must be a named dict or list of raster paths")
    return context


def read_context_stack(context, h, w, *, reference=None, expected=None,
                       window=None, ds_cache=None):
    named = _named_context(context)
    if expected is not None:
        names = expected.get("context_names", [])
        if set(named) != set(names):
            raise ValueError(f"Context names mismatch: expected {names}, got {list(named)}")
    else:
        names = list(named)
    arrays, counts = [], []
    x, y, bw, bh = window or (0, 0, w, h)
    for name in names:
        source = named[name]
        if isinstance(source, (str, os.PathLike)):
            info = get_image_info(source)
            if info[:2] != (h, w):
                raise ValueError(f"Context '{name}' dimensions differ from the image")
            if reference is not None:
                validate_grid(reference, info, bands=False, name=f"context '{name}'")
            arr = flat_block(source, x, y, bw, bh, ds_cache)
        elif isinstance(source, np.ndarray):
            arr = np.asarray(source, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2 or arr.shape[0] != h * w:
                raise ValueError(f"Context '{name}' must have shape (H*W, bands)")
            arr = arr.reshape(h, w, -1)[y:y + bh, x:x + bw].reshape(bh * bw, -1).copy()
            arr[~np.isfinite(arr)] = np.nan
        else:
            raise TypeError(f"Context '{name}' must be a raster path or numpy array")
        arrays.append(arr)
        counts.append(arr.shape[1])
    if expected is not None and counts != expected.get("context_band_counts", []):
        raise ValueError("Context band counts differ from the model's named feature schema")
    return (np.hstack(arrays) if arrays else None,
            {"context_names": names, "context_band_counts": counts,
             "total_context_bands": sum(counts)})


def write_geotiff(path, data, geo_transform, projection, descriptions=None, nodata=np.nan):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError("GeoTIFF data must have shape (H, W, bands)")
    h, w, count = data.shape
    ds = gdal.GetDriverByName("GTiff").Create(os.fspath(path), w, h, count, gdal.GDT_Float32,
                                            options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)
    fill = np.nan if nodata is None else nodata
    for i in range(count):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(np.where(np.isfinite(data[:, :, i]), data[:, :, i], fill))
        band.SetNoDataValue(float(fill))
        if descriptions and i < len(descriptions):
            band.SetDescription(descriptions[i])
    ds.FlushCache()
    ds = None
