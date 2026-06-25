import zarr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter

ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"

# ── load cube ──────────────────────────────────────────────────────────────────
store = zarr.open(ZARR_PATH, mode="r")

b04 = store["B04"][:]   # Red
b05 = store["B05"][:]   # Red-edge 1  ← primary CLR signal
b07 = store["B07"][:]   # Red-edge 3  ← secondary CLR signal
b08 = store["B08"][:]   # NIR
time_days = store["time"][:]

# convert time to readable dates
import pandas as pd
dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")
print(f"Date range: {dates[0].date()} → {dates[-1].date()}")
print(f"Total time steps: {len(dates)}")

# ── pick best time step (lowest cloud, most recent) ───────────────────────────
# n_obs tells us how many valid observations went into each pixel
n_obs = store["n_obs"][:]

# use the most recent time step with good coverage
coverage = (b04 > 0).sum(axis=(1, 2))  # count valid pixels per time step
best_t = int(np.argmax(coverage[-20:]) + len(coverage) - 20)  # last 20 steps
print(f"Using time step {best_t}: {dates[best_t].date()}")

# ── extract single time step ───────────────────────────────────────────────────
red     = b04[best_t].astype(float)
re1     = b05[best_t].astype(float)   # red-edge 1
re3     = b07[best_t].astype(float)   # red-edge 3
nir     = b08[best_t].astype(float)

# mask nodata
mask = (red > 0) & (nir > 0) & (re1 > 0)

# ── compute spectral indices ───────────────────────────────────────────────────
# NDVI — general vegetation health
ndvi = np.where(mask, (nir - red) / (nir + red + 1e-9), np.nan)

# Red-edge NDVI (RENDVI) — chlorophyll stress, primary CLR signal
rendvi = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)

# Chlorophyll Red-Edge Index (CIre) — sensitive to early stress
cire = np.where(mask, (re3 / re1) - 1, np.nan)

print(f"NDVI range:   {np.nanmin(ndvi):.3f} → {np.nanmax(ndvi):.3f}")
print(f"RENDVI range: {np.nanmin(rendvi):.3f} → {np.nanmax(rendvi):.3f}")
print(f"CIre range:   {np.nanmin(cire):.3f} → {np.nanmax(cire):.3f}")

# ── compute infection probability score (0–100) ────────────────────────────────
# Low NDVI + low RENDVI + low CIre = higher infection probability
# Normalise each index so low values → high risk score

def norm_inverse(arr, lo, hi):
    """Low values of arr → high score (0-1)"""
    clipped = np.clip(arr, lo, hi)
    return 1.0 - (clipped - lo) / (hi - lo + 1e-9)

ndvi_score   = norm_inverse(ndvi,   0.1, 0.8)   # healthy coffee ~0.6-0.8
rendvi_score = norm_inverse(rendvi, 0.0, 0.4)
cire_score   = norm_inverse(cire,   0.5, 3.0)

# weighted combination — red-edge bands weighted higher
prob = (
    0.25 * ndvi_score +
    0.40 * rendvi_score +
    0.35 * cire_score
) * 100

prob = np.where(mask, np.clip(prob, 0, 100), np.nan)

print(f"\nInfection probability range: {np.nanmin(prob):.1f}% → {np.nanmax(prob):.1f}%")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)
axes[0].set_title(f"NDVI — {dates[best_t].date()}")
axes[0].axis("off")

axes[1].imshow(rendvi, cmap="RdYlGn", vmin=0, vmax=0.5)
axes[1].set_title("Red-edge NDVI (CLR stress signal)")
axes[1].axis("off")

cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)
im = axes[2].imshow(prob, cmap=cmap_risk, vmin=0, vmax=100)
axes[2].set_title("Infection probability score (%)")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("stage1_infection_probability.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: stage1_infection_probability.png")