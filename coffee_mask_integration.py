import os
import zarr
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pickle
import affine

ZARR_PATH    = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
COFFEE_2020  = "coffee_probability_ethiopia_2020.tif"
COFFEE_2024  = "coffee_probability_ethiopia_2024.tif"

# ── load zarr grid ─────────────────────────────────────────────────────────────
print("Loading zarr grid...")
store     = zarr.open(ZARR_PATH, mode="r")
x_coords  = store["x"][:]
y_coords  = store["y"][:]
time_days = store["time"][:]
dates     = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")

target_shape     = (len(y_coords), len(x_coords))
target_transform = affine.Affine(10.0, 0.0, x_coords[0], 0.0, -10.0, y_coords[0])
target_crs       = "EPSG:32636"

def load_coffee_mask(path, target_shape, target_transform, target_crs, threshold=0.5):
    """Load and reproject coffee probability map to match Sentinel-2 grid"""
    with rasterio.open(path) as src:
        raw = np.array(src.read(1), dtype=float)
        out = np.zeros(target_shape, dtype=float)
        reproject(raw, out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=target_transform, dst_crs=target_crs,
                  resampling=Resampling.bilinear)
    out = np.where(out > 0, out, np.nan)
    mask = out > threshold
    print(f"  Coffee pixels (>{threshold}): {mask.sum():,}")
    print(f"  Coffee area: {mask.sum() * 100 / 10000:.1f} ha")
    print(f"  Coverage: {100 * mask.sum() / np.prod(target_shape):.1f}%")
    return out, mask

# ── load coffee masks ──────────────────────────────────────────────────────────
print("\nLoading coffee probability masks...")
print("2020:")
coffee_prob_2020, coffee_mask_2020 = load_coffee_mask(
    COFFEE_2020, target_shape, target_transform, target_crs)

if os.path.exists(COFFEE_2024):
    print("2024:")
    coffee_prob_2024, coffee_mask_2024 = load_coffee_mask(
        COFFEE_2024, target_shape, target_transform, target_crs)
else:
    print("2024 coffee mask not found — using 2020")
    coffee_prob_2024, coffee_mask_2024 = coffee_prob_2020, coffee_mask_2020

# ── load main model cache ──────────────────────────────────────────────────────
print("\nLoading main model cache...")
with open("clr_model_cache.pkl", "rb") as f:
    cache = pickle.load(f)

prob            = cache["prob"]
vegetation_mask = cache["vegetation_mask"]
seed_mask       = cache["seed_mask"]
spread_risk     = cache["spread_risk"]
ndvi            = cache["ndvi"]
rendvi          = cache["rendvi"]
best_date       = cache["best_date"]
pre_symptomatic = cache["pre_symptomatic"]
symptomatic     = cache["symptomatic"]

# ── apply coffee mask to probability map ──────────────────────────────────────
print("\nApplying coffee mask...")

# coffee-masked probability — only score actual coffee pixels
prob_coffee = np.where(coffee_mask_2024 & vegetation_mask, prob, np.nan)

# coffee-masked pre-symptomatic detection
pre_symp_coffee = pre_symptomatic & coffee_mask_2024
symptomatic_coffee = symptomatic & coffee_mask_2024
seed_coffee = seed_mask & coffee_mask_2024

# coffee-masked spread risk
spread_coffee = np.where(coffee_mask_2024 & ~seed_coffee, spread_risk, np.nan)

print(f"\n── Coffee-masked Results ────────────────────────────────────")
print(f"  Date analysed:                    {best_date}")
print(f"  Total coffee area detected:       {coffee_mask_2024.sum() * 100 / 10000:.1f} ha")
print(f"  Pre-symptomatic CLR (coffee):     {pre_symp_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  Symptomatic CLR (coffee):         {symptomatic_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  Confirmed infection (coffee):     {seed_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  High spread risk (coffee, >70%):  "
      f"{((spread_coffee >= 70) & coffee_mask_2024).sum() * 100 / 10000:.2f} ha")
print(f"  Moderate spread risk (40-70%):    "
      f"{((spread_coffee >= 40) & (spread_coffee < 70) & coffee_mask_2024).sum() * 100 / 10000:.2f} ha")
print("────────────────────────────────────────────────────────────")

# compare before and after masking
print(f"\n── Before vs After Coffee Mask ──────────────────────────────")
print(f"  {'Metric':<30} {'Before':>10} {'After':>10}")
print(f"  {'-'*52}")
print(f"  {'Pre-symptomatic area':<30} "
      f"{pre_symptomatic.sum() * 100 / 10000:>9.2f}ha "
      f"{pre_symp_coffee.sum() * 100 / 10000:>9.2f}ha")
print(f"  {'Symptomatic area':<30} "
      f"{symptomatic.sum() * 100 / 10000:>9.2f}ha "
      f"{symptomatic_coffee.sum() * 100 / 10000:>9.2f}ha")
print(f"  {'Confirmed infection':<30} "
      f"{seed_mask.sum() * 100 / 10000:>9.2f}ha "
      f"{seed_coffee.sum() * 100 / 10000:>9.2f}ha")
print("────────────────────────────────────────────────────────────")

# ── plots ──────────────────────────────────────────────────────────────────────
cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)
cmap_coffee = mcolors.LinearSegmentedColormap.from_list(
    "coffee", ["white", "#C8A96E", "#6B3A2A"]
)
cmap_spread = mcolors.LinearSegmentedColormap.from_list(
    "spread", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A"]
)
cmap_stage = mcolors.ListedColormap(["#5DCAA5", "#EF9F27", "#A32D2D"])

# ── MAP 16 — coffee probability + masked CLR detection ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    f"Coffee-masked CLR detection — {best_date}\n"
    "Only pixels confirmed as coffee by Forest Data Partnership model",
    fontsize=14, fontweight="bold"
)

# top left — coffee probability
im0 = axes[0, 0].imshow(coffee_prob_2024, cmap=cmap_coffee, vmin=0, vmax=1)
axes[0, 0].set_title("Coffee probability\n(Forest Data Partnership model 2025b)")
axes[0, 0].axis("off")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04,
             label="Probability (0=not coffee, 1=coffee)")

# top right — coffee-masked infection probability
im1 = axes[0, 1].imshow(prob_coffee, cmap=cmap_risk, vmin=0, vmax=100)
axes[0, 1].set_title("CLR infection probability\n(coffee pixels only)")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="%")

# bottom left — CLR stage map (coffee only)
stage_map = np.where(coffee_mask_2024 & vegetation_mask, 0, np.nan)
stage_map = np.where(pre_symp_coffee, 1, stage_map)
stage_map = np.where(symptomatic_coffee, 2, stage_map)

bounds     = [-0.5, 0.5, 1.5, 2.5]
norm_stage = mcolors.BoundaryNorm(bounds, cmap_stage.N)
im2 = axes[1, 0].imshow(stage_map, cmap=cmap_stage, norm=norm_stage)
cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
cbar2.ax.set_yticklabels([
    "Healthy coffee",
    f"Pre-symptomatic\n({pre_symp_coffee.sum() * 100 / 10000:.2f} ha)",
    f"Symptomatic\n({symptomatic_coffee.sum() * 100 / 10000:.2f} ha)"
])
axes[1, 0].set_title("CLR detection stages\n(coffee pixels only)")
axes[1, 0].axis("off")

# bottom right — spread risk on coffee pixels
im3 = axes[1, 1].imshow(spread_coffee, cmap=cmap_spread, vmin=0, vmax=100)
axes[1, 1].imshow(
    np.where(seed_coffee, 1, np.nan),
    cmap=mcolors.ListedColormap(["#A32D2D"]),
    vmin=0, vmax=1
)
axes[1, 1].set_title("Spread risk — coffee pixels only\n(dark red = confirmed source)")
axes[1, 1].axis("off")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Spread risk %")

# add stats box
stats_text = (
    f"Coffee area: {coffee_mask_2024.sum() * 100 / 10000:.0f} ha\n"
    f"Pre-symptomatic: {pre_symp_coffee.sum() * 100 / 10000:.2f} ha\n"
    f"Symptomatic: {symptomatic_coffee.sum() * 100 / 10000:.2f} ha\n"
    f"Confirmed: {seed_coffee.sum() * 100 / 10000:.2f} ha"
)
fig.text(0.02, 0.02, stats_text, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("map16_coffee_masked_clr.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: map16_coffee_masked_clr.png")

# ── MAP 17 — side by side before vs after coffee mask ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(
    "Before vs after coffee mask\nLeft: all vegetation  |  Right: confirmed coffee only",
    fontsize=13, fontweight="bold"
)

im_before = axes[0].imshow(prob, cmap=cmap_risk, vmin=0, vmax=100)
axes[0].set_title(
    f"All vegetation\n"
    f"Pre-symptomatic: {pre_symptomatic.sum() * 100 / 10000:.2f} ha"
)
axes[0].axis("off")
plt.colorbar(im_before, ax=axes[0], fraction=0.046, pad=0.04, label="%")

im_after = axes[1].imshow(prob_coffee, cmap=cmap_risk, vmin=0, vmax=100)
axes[1].set_title(
    f"Coffee pixels only\n"
    f"Pre-symptomatic: {pre_symp_coffee.sum() * 100 / 10000:.2f} ha"
)
axes[1].axis("off")
plt.colorbar(im_after, ax=axes[1], fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("map17_before_after_coffee_mask.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map17_before_after_coffee_mask.png")

# ── update cache with coffee-masked results ────────────────────────────────────
cache["prob_coffee"]         = prob_coffee
cache["coffee_mask"]         = coffee_mask_2024
cache["pre_symp_coffee"]     = pre_symp_coffee
cache["symptomatic_coffee"]  = symptomatic_coffee
cache["seed_coffee"]         = seed_coffee
cache["spread_coffee"]       = spread_coffee

with open("clr_model_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print("Cache updated with coffee mask")

print("\n── Final pitch numbers ─────────────────────────────────────")
print(f"  Total confirmed coffee area:      {coffee_mask_2024.sum() * 100 / 10000:.0f} ha")
print(f"  Pre-symptomatic CLR in coffee:    {pre_symp_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  Symptomatic CLR in coffee:        {symptomatic_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  Confirmed infection in coffee:    {seed_coffee.sum() * 100 / 10000:.2f} ha")
print(f"  Model validation (LOO CV):        r = 0.776, p = 0.0002")
print(f"  Seasonal fingerprint:             p = 0.0000")
print(f"  Outbreak detection:               March 2021 peak")
print(f"  Drought ruled out:                2020 wettest year (flooding confirmed)")
print("────────────────────────────────────────────────────────────")