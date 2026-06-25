import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pickle
from scipy.stats import pearsonr

SPOT_2020 = "F1-coffee-leaf-rust-ethiopia/SPOT-HighRes-Imagery/spot6_pms_2020-12-12.tiff"
SPOT_2024 = "F1-coffee-leaf-rust-ethiopia/SPOT-HighRes-Imagery/spot6_pms_2024-11-18.tiff"
GEOJSON   = "F1-coffee-leaf-rust-ethiopia/F1_CLR Survey Farm Level.geojson"

# ── load SPOT imagery ──────────────────────────────────────────────────────────
# SPOT 6 PMS band order: Blue, Green, Red, NIR
print("Loading SPOT imagery...")

def load_spot(path):
    with rasterio.open(path) as src:
        blue = src.read(1).astype(float)
        green = src.read(2).astype(float)
        red  = src.read(3).astype(float)
        nir  = src.read(4).astype(float)
        transform = src.transform
        crs = src.crs
    return blue, green, red, nir, transform, crs

blue20, green20, red20, nir20, transform, crs = load_spot(SPOT_2020)
blue24, green24, red24, nir24, _, _           = load_spot(SPOT_2024)

print(f"  Shape: {red20.shape}")
print(f"  Resolution: 1.5m")
print(f"  Area: {red20.shape[0] * red20.shape[1] * 1.5 * 1.5 / 10000:.0f} ha")

# ── mask nodata ────────────────────────────────────────────────────────────────
mask20 = (red20 > 0) & (nir20 > 0) & (blue20 > 0)
mask24 = (red24 > 0) & (nir24 > 0) & (blue24 > 0)

# ── compute indices at 1.5m ────────────────────────────────────────────────────
print("Computing 1.5m indices...")

with np.errstate(invalid="ignore", divide="ignore"):
    # 2020
    ndvi20  = np.where(mask20, (nir20 - red20) / (nir20 + red20 + 1e-9), np.nan)
    evi20   = np.where(mask20,
                2.5 * (nir20 - red20) / (nir20 + 6*red20 - 7.5*blue20 + 1 + 1e-9),
                np.nan)

    # 2024
    ndvi24  = np.where(mask24, (nir24 - red24) / (nir24 + red24 + 1e-9), np.nan)
    evi24   = np.where(mask24,
                2.5 * (nir24 - red24) / (nir24 + 6*red24 - 7.5*blue24 + 1 + 1e-9),
                np.nan)

# NDVI change 2020 → 2024
ndvi_change = np.where(mask20 & mask24, ndvi24 - ndvi20, np.nan)

print(f"  NDVI 2020: {np.nanmin(ndvi20):.3f} → {np.nanmax(ndvi20):.3f}")
print(f"  NDVI 2024: {np.nanmin(ndvi24):.3f} → {np.nanmax(ndvi24):.3f}")
print(f"  NDVI change: {np.nanmin(ndvi_change):.3f} → {np.nanmax(ndvi_change):.3f}")
print(f"  Mean NDVI change: {np.nanmean(ndvi_change):+.3f}")

# ── canopy classification at 1.5m ─────────────────────────────────────────────
# dense canopy: high NDVI + high NIR
# coffee-likely understory gaps: lower NDVI patches within forest
dense_canopy_24  = mask24 & (ndvi24 > 0.5)
moderate_veg_24  = mask24 & (ndvi24 > 0.3) & (ndvi24 <= 0.5)
sparse_veg_24    = mask24 & (ndvi24 > 0.1) & (ndvi24 <= 0.3)

print(f"\n  2024 canopy classification:")
print(f"  Dense canopy (NDVI>0.5):    {dense_canopy_24.sum() * 1.5**2 / 10000:.1f} ha")
print(f"  Moderate veg (0.3-0.5):     {moderate_veg_24.sum() * 1.5**2 / 10000:.1f} ha")
print(f"  Sparse veg (0.1-0.3):       {sparse_veg_24.sum() * 1.5**2 / 10000:.1f} ha")

# ── canopy stress — areas where NDVI declined 2020→2024 ───────────────────────
canopy_stress = (
    mask20 & mask24 &
    (ndvi20 > 0.4) &        # was healthy canopy in 2020
    (ndvi_change < -0.05)   # declined by more than 0.05
)

canopy_gain = (
    mask20 & mask24 &
    (ndvi24 > 0.4) &
    (ndvi_change > 0.05)
)

print(f"\n  Canopy stress (NDVI declined >0.05): {canopy_stress.sum() * 1.5**2 / 10000:.1f} ha")
print(f"  Canopy gain  (NDVI gained  >0.05):   {canopy_gain.sum() * 1.5**2 / 10000:.1f} ha")

# ── load ground truth farms ────────────────────────────────────────────────────
print("\nOverlaying ground truth farm locations...")
farms = gpd.read_file(GEOJSON).to_crs(crs)

# extract SPOT NDVI at each farm location
def get_pixel(arr, transform, x, y):
    """Get pixel value at UTM coordinates"""
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    col = np.clip(col, 0, arr.shape[1]-1)
    row = np.clip(row, 0, arr.shape[0]-1)
    # 5x5 neighbourhood at 1.5m = ~7.5m patch
    r0 = max(0, row-2); r1 = min(arr.shape[0], row+3)
    c0 = max(0, col-2); c1 = min(arr.shape[1], col+3)
    return float(np.nanmean(arr[r0:r1, c0:c1]))

farm_data = []
for _, farm in farms.iterrows():
    x, y = farm.geometry.x, farm.geometry.y
    ndvi20_val    = get_pixel(ndvi20,      transform, x, y)
    ndvi24_val    = get_pixel(ndvi24,      transform, x, y)
    change_val    = get_pixel(ndvi_change, transform, x, y)
    stress_val    = get_pixel(
        np.where(mask24, ndvi_change, np.nan), transform, x, y)

    farm_data.append({
        "farm":      int(farm["farm"]),
        "inc":       float(farm["inc"]),
        "sev":       float(farm["sev"]),
        "altitude":  float(farm["altitude"]),
        "ndvi_2020": ndvi20_val,
        "ndvi_2024": ndvi24_val,
        "ndvi_change": change_val,
        "x": x, "y": y
    })

import pandas as pd
df = pd.DataFrame(farm_data).dropna()
print(f"  {len(df)} farms with SPOT data")

if len(df) >= 5:
    r_ndvi20, p20 = pearsonr(df["ndvi_2020"],    df["inc"])
    r_ndvi24, p24 = pearsonr(df["ndvi_2024"],    df["inc"])
    r_change, pc  = pearsonr(df["ndvi_change"],  df["inc"])

    print(f"\n── SPOT vs CLR field incidence ─────────────────────────────")
    print(f"  SPOT NDVI 2020 vs incidence:    r = {r_ndvi20:.3f}  p = {p20:.4f}")
    print(f"  SPOT NDVI 2024 vs incidence:    r = {r_ndvi24:.3f}  p = {p24:.4f}")
    print(f"  NDVI change 2020→2024 vs inc:   r = {r_change:.3f}  p = {pc:.4f}")

    if r_change < -0.2:
        print(f"  ✓ Areas with more NDVI decline show higher CLR incidence")
        print(f"    Canopy stress correlates with under-canopy CLR")
    print("────────────────────────────────────────────────────────────")

    print(f"\n── Farm SPOT data ───────────────────────────────────────────")
    print(f"  {'Farm':<6} {'Alt':>6} {'Inc':>6} {'NDVI20':>8} {'NDVI24':>8} {'Change':>8}")
    print("  " + "-"*46)
    for _, row in df.sort_values("inc", ascending=False).iterrows():
        print(f"  {int(row['farm']):<6} {row['altitude']:>5.0f}m "
              f"{row['inc']:>5.1f}% {row['ndvi_2020']:>8.3f} "
              f"{row['ndvi_2024']:>8.3f} {row['ndvi_change']:>+8.3f}")
    print("────────────────────────────────────────────────────────────")

# ── plots ──────────────────────────────────────────────────────────────────────
print("\nGenerating SPOT maps...")

def normalise_band(arr, lo_pct=2, hi_pct=98):
    valid = arr[arr > 0]
    lo = np.percentile(valid, lo_pct)
    hi = np.percentile(valid, hi_pct)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

# ── MAP A — SPOT true colour 2020 vs 2024 ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle(
    "SPOT 6 — 1.5m resolution true colour\n"
    "Left: Dec 2020  |  Right: Nov 2024",
    fontsize=14, fontweight="bold"
)

rgb20 = np.dstack([
    normalise_band(red20),
    normalise_band(green20),
    normalise_band(blue20)
])
rgb24 = np.dstack([
    normalise_band(red24),
    normalise_band(green24),
    normalise_band(blue24)
])

axes[0].imshow(rgb20)
axes[0].set_title("SPOT 6 — Dec 2020\n1.5m resolution — individual tree crowns visible")
axes[0].axis("off")

# overlay farm points
for _, farm in farms.iterrows():
    x, y = farm.geometry.x, farm.geometry.y
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    if 0 <= col < red20.shape[1] and 0 <= row < red20.shape[0]:
        color = "#A32D2D" if farm["inc"] > 60 else "#EF9F27" if farm["inc"] > 40 else "#5DCAA5"
        axes[0].plot(col, row, "o", color=color, markersize=8,
                     markeredgecolor="white", markeredgewidth=1.5)

axes[1].imshow(rgb24)
axes[1].set_title("SPOT 6 — Nov 2024\n1.5m resolution — individual tree crowns visible")
axes[1].axis("off")

for _, farm in farms.iterrows():
    x, y = farm.geometry.x, farm.geometry.y
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    if 0 <= col < red24.shape[1] and 0 <= row < red24.shape[0]:
        color = "#A32D2D" if farm["inc"] > 60 else "#EF9F27" if farm["inc"] > 40 else "#5DCAA5"
        axes[1].plot(col, row, "o", color=color, markersize=8,
                     markeredgecolor="white", markeredgewidth=1.5,
                     label=f"Farm (inc={farm['inc']:.0f}%)")

from matplotlib.lines import Line2D
legend = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#A32D2D",
           markersize=8, label="High incidence (>60%)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#EF9F27",
           markersize=8, label="Medium incidence (40-60%)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#5DCAA5",
           markersize=8, label="Low incidence (<40%)"),
]
axes[1].legend(handles=legend, loc="lower right", fontsize=9,
               facecolor="white", framealpha=0.8)

plt.tight_layout()
plt.savefig("map22_spot_truecolour.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map22_spot_truecolour.png")

# ── MAP B — NDVI change 2020→2024 + canopy stress ─────────────────────────────
cmap_change = mcolors.LinearSegmentedColormap.from_list(
    "change", ["#A32D2D", "#EF9F27", "#F1EFE8", "#5DCAA5", "#085041"]
)
cmap_ndvi = mcolors.LinearSegmentedColormap.from_list(
    "ndvi", ["#A32D2D", "#EF9F27", "#C0DD97", "#5DCAA5", "#085041"]
)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle(
    "SPOT 6 canopy analysis — 1.5m resolution\n"
    "Canopy NDVI change 2020→2024 as proxy for under-canopy CLR stress",
    fontsize=13, fontweight="bold"
)

im0 = axes[0].imshow(ndvi20, cmap=cmap_ndvi, vmin=0, vmax=0.8)
axes[0].set_title("NDVI Dec 2020\n(pre-survey period)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(ndvi24, cmap=cmap_ndvi, vmin=0, vmax=0.8)
axes[1].set_title("NDVI Nov 2024")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(ndvi_change, cmap=cmap_change, vmin=-0.3, vmax=0.3)
axes[2].set_title(
    f"NDVI change 2020→2024\n"
    f"Red = canopy declined  |  Stress: {canopy_stress.sum() * 1.5**2 / 10000:.0f} ha"
)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ΔNDVI")

# overlay farm points on change map
for _, row_d in df.iterrows():
    x, y = row_d["x"], row_d["y"]
    col = int((x - transform.c) / transform.a)
    row_px = int((y - transform.f) / transform.e)
    if 0 <= col < ndvi_change.shape[1] and 0 <= row_px < ndvi_change.shape[0]:
        size = 6 + row_d["inc"] / 10
        axes[2].plot(col, row_px, "o", markersize=size,
                     markerfacecolor="white", markeredgecolor="black",
                     markeredgewidth=1.5, zorder=5)
        axes[2].annotate(f"{row_d['inc']:.0f}%",
                         (col, row_px), textcoords="offset points",
                         xytext=(5, 5), fontsize=7, color="white",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="black", alpha=0.6))

plt.tight_layout()
plt.savefig("map23_spot_canopy_change.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map23_spot_canopy_change.png")

# ── MAP C — zoom into high incidence farm area ─────────────────────────────────
# zoom to farm cluster at 1100m (highest incidence)
high_inc_farms = df[df["inc"] > 70]
if len(high_inc_farms) > 0:
    cx = high_inc_farms["x"].mean()
    cy = high_inc_farms["y"].mean()

    # convert to pixel
    cc = int((cx - transform.c) / transform.a)
    cr = int((cy - transform.f) / transform.e)

    # 500 pixel window = 750m at 1.5m resolution
    w = 500
    r0 = max(0, cr-w); r1 = min(rgb24.shape[0], cr+w)
    c0 = max(0, cc-w); c1 = min(rgb24.shape[1], cc+w)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(
        "SPOT 6 zoom — high incidence farm cluster (>70% CLR)\n"
        "1.5m resolution — 750m × 750m area",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(rgb20[r0:r1, c0:c1])
    axes[0].set_title("True colour Dec 2020")
    axes[0].axis("off")

    axes[1].imshow(rgb24[r0:r1, c0:c1])
    axes[1].set_title("True colour Nov 2024")
    axes[1].axis("off")

    im3 = axes[2].imshow(ndvi_change[r0:r1, c0:c1],
                          cmap=cmap_change, vmin=-0.3, vmax=0.3)
    axes[2].set_title("NDVI change 2020→2024\nRed = canopy declined")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="ΔNDVI")

    # mark farm locations in zoom
    for _, row_d in high_inc_farms.iterrows():
        fc = int((row_d["x"] - transform.c) / transform.a) - c0
        fr = int((row_d["y"] - transform.f) / transform.e) - r0
        for ax in axes:
            ax.plot(fc, fr, "o", markersize=12,
                    markerfacecolor="#A32D2D", markeredgecolor="white",
                    markeredgewidth=2, zorder=5)
            ax.annotate(f"Farm {int(row_d['farm'])}\n{row_d['inc']:.0f}% inc",
                        (fc, fr), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#A32D2D", alpha=0.8))

    plt.tight_layout()
    plt.savefig("map24_spot_zoom_highincidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: map24_spot_zoom_highincidence.png")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n── SPOT Analysis Summary ────────────────────────────────────")
print(f"  Resolution:              1.5m (vs 10m Sentinel-2)")
print(f"  Dates:                   Dec 2020, Nov 2024")
print(f"  Area covered:            {red20.shape[0]*red20.shape[1]*1.5**2/10000:.0f} ha")
print(f"  Dense canopy 2024:       {dense_canopy_24.sum()*1.5**2/10000:.0f} ha")
print(f"  Canopy stress 2020→2024: {canopy_stress.sum()*1.5**2/10000:.0f} ha")
print(f"  Mean NDVI change:        {np.nanmean(ndvi_change):+.3f}")
if len(df) >= 5:
    print(f"\n  SPOT NDVI 2020 vs CLR incidence: r = {r_ndvi20:.3f}")
    print(f"  SPOT NDVI 2024 vs CLR incidence: r = {r_ndvi24:.3f}")
    print(f"  NDVI change vs CLR incidence:    r = {r_change:.3f}")
print()
print("  At 1.5m individual tree crowns are visible")
print("  Canopy gaps and stressed crowns detectable")
print("  Next step: train crown-level classifier to detect")
print("  canopy stress adjacent to known CLR infection points")
print("────────────────────────────────────────────────────────────")