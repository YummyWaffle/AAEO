# -*- coding: utf-8 -*-
"""
Visualization helpers for TLG anomaly detection results.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def anomaly_preview(
    anomaly_map: np.ndarray,
    output_path: str,
    background_path: str = None,
    dpi: int = 150,
    title: str = None,
) -> None:
    """
    Save a JPG visualization of an anomaly score map.

    Parameters
    ----------
    anomaly_map : np.ndarray
        2-D float32 array (H, W) of anomaly scores in [0, 1].
    output_path : str
        Output JPG path (e.g. ``result_preview.jpg``).
    background_path : str, optional
        Path to a GeoTIFF to use as the background image.  The first three
        bands are treated as RGB.  If not given, the anomaly map is shown
        on its own.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    """
    if background_path is not None:
        _overlay_plot(anomaly_map, background_path, output_path, dpi, title)
    else:
        _heatmap_plot(anomaly_map, output_path, dpi, title)


# ---------------------------------------------------------------------------
# Internal plotters
# ---------------------------------------------------------------------------

def _stretch_band(band: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Percentile-stretch a single band to [0, 1] for display."""
    valid = band[~np.isnan(band)]
    if len(valid) == 0:
        return np.zeros_like(band)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((band - lo) / (hi - lo), 0, 1)


def _overlay_plot(anomaly_map, background_path, output_path, dpi, title):
    """Anomaly heatmap overlaid on an RGB background."""
    from ._io import read_image_full

    bg_flat, h, w, c, _, _ = read_image_full(background_path)
    bg_cube = bg_flat.reshape(h, w, c)

    # Build an RGB composite from the first 3 bands (or fewer)
    rgb_bands = min(c, 3)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(rgb_bands):
        rgb[:, :, i] = _stretch_band(bg_cube[:, :, i])

    # If fewer than 3 bands, replicate the first band
    if rgb_bands < 3:
        for i in range(rgb_bands, 3):
            rgb[:, :, i] = rgb[:, :, 0]

    # 0.0–0.2: transparent dark gray (no hue, just subtle darkening).
    # 0.2–1.0: jet colormap (cold → warm, scores drive hue and visibility).
    anomaly_valid = np.nan_to_num(anomaly_map, nan=0.0)
    cmap = plt.cm.jet
    overlay_rgba = cmap(anomaly_valid)
    low = anomaly_valid < 0.2
    overlay_rgba[low, :3] = 0.05  # near-black, no colour signal
    alpha = np.where(np.isnan(anomaly_map), 0.0, 0.30 + anomaly_valid * 0.35)

    blended = rgb * (1 - alpha[:, :, None]) + overlay_rgba[:, :, :3] * alpha[:, :, None]
    blended = np.clip(blended, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(blended)
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Anomaly Score", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _heatmap_plot(anomaly_map, output_path, dpi, title):
    """Standalone anomaly heatmap with colorbar."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        anomaly_map,
        cmap="jet",
        norm=Normalize(vmin=0, vmax=1),
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Anomaly Score", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
