import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift

ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"

# ── load cube ──────────────────────────────────────────────────────────────────
print("Loading data...")
store = zarr.open(ZARR_PATH, mode="r")

b04 = store["B04"][:]
b05 = store["B05"][:]
b07 = store["B07"][:]
b08 = store["B08"][:]
time_days = store["time"][:]

dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")
print(f"Date range: {dates[0].date()} → {dates[-1].date()}")
print(f"Total time steps: {len(dates)}")

# ── pick best time step ────────────────────────────────────────────────────────
coverage = (b04 > 0).sum(axis=(1, 2))
best_t = int(np.argmax(coverage[-20:]) + len(coverage) - 20)
print(f"Using time step {best_t}: {dates[best_t].date()}")

# ── extract bands ──────────────────────────────────────────────────────────────
red  = b04[best_t].astype(float)
re1  = b05[best_t].astype(float)
re3  = b07[best_t].astype(float)
nir  = b08[best_t].astype(float)
mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0)

# ── spectral indices ───────────────────────────────────────────────────────────
with np.errstate(invalid="ignore", divide="ignore"):
    ndvi   = np.where(mask, (nir - red) / (nir + red + 1e-9), np.nan)
    rendvi = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)
    cire   = np.where(mask, (re3 / re1) - 1,                   np.nan)

print(f"NDVI range:   {np.nanmin(ndvi):.3f} → {np.nanmax(ndvi):.3f}")
print(f"RENDVI range: {np.nanmin(rendvi):.3f} → {np.nanmax(rendvi):.3f}")
print(f"CIre range:   {np.nanmin(cire):.3f} → {np.nanmax(cire):.3f}")

# ── vegetation mask — exclude bare soil, roads, buildings ─────────────────────
vegetation_mask = mask & (ndvi > 0.2) & (ndvi < 0.85)
print(f"Vegetation pixels: {vegetation_mask.sum():,} of {mask.sum():,} valid pixels")

# ── stage 1: infection probability ────────────────────────────────────────────
print("\nComputing infection probability...")

def norm_inverse(arr, lo, hi):
    clipped = np.clip(arr, lo, hi)
    return 1.0 - (clipped - lo) / (hi - lo + 1e-9)

ndvi_score   = norm_inverse(ndvi,   0.2, 0.8)
rendvi_score = norm_inverse(rendvi, 0.0, 0.4)
cire_score   = norm_inverse(cire,   0.5, 3.0)

prob = (
    0.25 * ndvi_score +
    0.40 * rendvi_score +
    0.35 * cire_score
) * 100

prob = np.where(vegetation_mask, np.clip(prob, 0, 100), np.nan)
print(f"Infection probability range: {np.nanmin(prob):.1f}% → {np.nanmax(prob):.1f}%")

# ── stage 2: spread prediction ─────────────────────────────────────────────────
print("\nComputing spread risk...")

CONFIRMED_THRESHOLD = 70
seed_mask = (prob >= CONFIRMED_THRESHOLD) & vegetation_mask
print(f"Confirmed/high infection pixels: {seed_mask.sum():,}")
print(f"As % of vegetation area: {100 * seed_mask.sum() / vegetation_mask.sum():.1f}%")

# distance from nearest infection source — normalise BEFORE masking
dist_from_infection = distance_transform_edt(~seed_mask).astype(float)
max_dist = float(np.percentile(dist_from_infection, 95))
proximity_score = 1.0 - np.clip(dist_from_infection / max_dist, 0, 1)

# wind bias — SE direction
wind_biased = shift(proximity_score, shift=[8, 8], cval=0.0)

# smooth broadly so risk radiates as gradient
spread_risk_raw = gaussian_filter(wind_biased, sigma=25)

# normalise to 0-100 across full image first, then mask
s_min = spread_risk_raw.min()
s_max = spread_risk_raw.max()
spread_risk = 100 * (spread_risk_raw - s_min) / (s_max - s_min + 1e-9)

# apply mask — healthy vegetation pixels only
spread_risk = np.where(vegetation_mask & ~seed_mask, spread_risk, np.nan)

print(f"Spread risk range: {np.nanmin(spread_risk):.1f}% → {np.nanmax(spread_risk):.1f}%")

# ── plot ───────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)
cmap_spread = mcolors.LinearSegmentedColormap.from_list(
    "spread", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A"]
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Coffee Leaf Rust Analysis — {dates[best_t].date()}", fontsize=14, fontweight="bold")

im0 = axes[0, 0].imshow(ndvi, cmap="RdYlGn", vmin=0.1, vmax=0.85)
axes[0, 0].set_title("NDVI — vegetation health")
axes[0, 0].axis("off")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(rendvi, cmap="RdYlGn", vmin=0, vmax=0.5)
axes[0, 1].set_title("Red-edge NDVI — CLR stress signal")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[1, 0].imshow(prob, cmap=cmap_risk, vmin=0, vmax=100)
axes[1, 0].set_title("Stage 1 — infection probability (vegetation only, %)")
axes[1, 0].axis("off")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="%")

im3 = axes[1, 1].imshow(spread_risk, cmap=cmap_spread, vmin=0, vmax=100)
axes[1, 1].imshow(
    np.where(seed_mask, 1, np.nan),
    cmap=mcolors.ListedColormap(["#A32D2D"]),
    vmin=0, vmax=1
)
axes[1, 1].set_title("Stage 2 — spread risk\n(dark red = confirmed infection source)")
axes[1, 1].axis("off")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Spread risk %")

plt.tight_layout()
plt.savefig("clr_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: clr_analysis.png")

# ── farmer summary ─────────────────────────────────────────────────────────────
high_spread     = (spread_risk >= 70) & vegetation_mask & ~seed_mask
moderate_spread = (spread_risk >= 40) & (spread_risk < 70) & vegetation_mask & ~seed_mask
low_spread      = (spread_risk  < 40) & vegetation_mask & ~seed_mask

print("\n── Farm Risk Summary ──────────────────────────────────────")
print(f"  Date analysed:                  {dates[best_t].date()}")
print(f"  Total vegetation area:          {vegetation_mask.sum() * 100 / 10000:.1f} ha")
print(f"  Confirmed/high infection:       {seed_mask.sum() * 100 / 10000:.2f} ha")
print(f"  High spread risk   (>70%):      {high_spread.sum() * 100 / 10000:.2f} ha")
print(f"  Moderate spread risk (40-70%):  {moderate_spread.sum() * 100 / 10000:.2f} ha")
print(f"  Low spread risk    (<40%):      {low_spread.sum() * 100 / 10000:.2f} ha")
print("────────────────────────────────────────────────────────────")
print("  Priority: inspect and treat HIGH spread risk zones first")
print("────────────────────────────────────────────────────────────")