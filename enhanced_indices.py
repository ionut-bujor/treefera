import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pickle
import os
from scipy.stats import pearsonr
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift

ZARR_PATH  = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
MODEL_CACHE = "clr_model_cache.pkl"

print("Loading zarr...")
store     = zarr.open(ZARR_PATH, mode="r")
b02       = store["B02"][:]
b03       = store["B03"][:]
b04       = store["B04"][:]
b05       = store["B05"][:]
b07       = store["B07"][:]
b08       = store["B08"][:]
time_days = store["time"][:]
x_coords  = store["x"][:]
y_coords  = store["y"][:]
dates     = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")

coverage = (b04 > 0).sum(axis=(1, 2))
best_t   = int(np.argmax(coverage[-20:]) + len(coverage) - 20)
best_date = str(dates[best_t].date())
print(f"Using time step {best_t}: {best_date}")

# ── extract bands ──────────────────────────────────────────────────────────────
blue = b02[best_t].astype(float) / 10000
grn  = b03[best_t].astype(float) / 10000
red  = b04[best_t].astype(float) / 10000
re1  = b05[best_t].astype(float) / 10000
re3  = b07[best_t].astype(float) / 10000
nir  = b08[best_t].astype(float) / 10000
mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0) & (blue > 0)

# ── compute all indices ────────────────────────────────────────────────────────
print("Computing crop health indices...")

with np.errstate(invalid="ignore", divide="ignore"):

    # existing indices
    ndvi   = np.where(mask, (nir - red)   / (nir + red   + 1e-9), np.nan)
    rendvi = np.where(mask, (re3 - re1)   / (re3 + re1   + 1e-9), np.nan)
    cire   = np.where(mask, (re3 / re1) - 1,                       np.nan)

    # new indices
    # NDRE — normalised difference red edge
    # more targeted chlorophyll than NDVI, drops before NDVI does
    ndre = np.where(mask, (nir - re1) / (nir + re1 + 1e-9), np.nan)

    # EVI — enhanced vegetation index
    # less affected by canopy background, better for dense forest
    evi = np.where(mask,
        2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-9),
        np.nan)

    # SAVI — soil adjusted vegetation index
    # reduces soil noise where canopy is incomplete
    savi = np.where(mask,
        1.5 * (nir - red) / (nir + red + 0.5 + 1e-9),
        np.nan)

    # MCARI — modified chlorophyll absorption ratio index
    # specifically detects chlorophyll absorption — directly relevant to CLR
    # drops in first 10 days of infection
    mcari = np.where(mask,
        ((re1 - red) - 0.2 * (re1 - grn)) * (re1 / (red + 1e-9)),
        np.nan)

    # LAI proxy — leaf area index
    # CLR causes defoliation so LAI drops before visible yellowing
    lai = np.where(mask & (evi > 0),
        3.618 * evi - 0.118,
        np.nan)

# vegetation mask
vegetation_mask = mask & (ndvi > 0.2) & (ndvi < 0.85)
print(f"Vegetation pixels: {vegetation_mask.sum():,}")

# print index ranges
for name, arr in [("NDVI", ndvi), ("RENDVI", rendvi), ("CIre", cire),
                   ("NDRE", ndre), ("EVI", evi), ("SAVI", savi),
                   ("MCARI", mcari), ("LAI", lai)]:
    print(f"  {name:<8} {np.nanmin(arr):.3f} → {np.nanmax(arr):.3f}")

# ── norm inverse helper ────────────────────────────────────────────────────────
def ni(arr, lo, hi):
    return 1.0 - (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-9)

# ── stage 1: enhanced infection probability ────────────────────────────────────
print("\nComputing enhanced infection probability...")

# original 3-index model
prob_original = (
    0.25 * ni(ndvi,   0.2, 0.8) +
    0.40 * ni(rendvi, 0.0, 0.4) +
    0.35 * ni(cire,   0.5, 3.0)
) * 100

# enhanced 7-index model
# weights based on CLR sensitivity:
# red-edge indices highest — most specific to chlorophyll breakdown
# MCARI second — directly measures chlorophyll absorption
# NDRE third — chlorophyll content
# EVI/SAVI — general canopy health
# NDVI lowest — least specific
prob_enhanced = (
    0.20 * ni(ndvi,   0.2,  0.8)  +   # general health
    0.20 * ni(rendvi, 0.0,  0.4)  +   # red-edge stress
    0.15 * ni(cire,   0.5,  3.0)  +   # chlorophyll index
    0.15 * ni(ndre,   0.3,  0.6)  +   # red-edge chlorophyll
    0.15 * ni(mcari,  0.05, 0.8)  +   # chlorophyll absorption
    0.10 * ni(evi,    0.2,  0.7)  +   # canopy density
    0.05 * ni(savi,   0.2,  0.7)       # soil-adjusted
) * 100

prob_original = np.where(vegetation_mask, np.clip(prob_original, 0, 100), np.nan)
prob_enhanced = np.where(vegetation_mask, np.clip(prob_enhanced, 0, 100), np.nan)

print(f"  Original (3 indices):  {np.nanmin(prob_original):.1f}% → {np.nanmax(prob_original):.1f}%")
print(f"  Enhanced (7 indices):  {np.nanmin(prob_enhanced):.1f}% → {np.nanmax(prob_enhanced):.1f}%")

# ── pre-symptomatic detection ──────────────────────────────────────────────────
# pre-symptomatic: red-edge AND MCARI stressed but NDVI still healthy
# this is the key improvement — MCARI drops before NDVI
pre_symptomatic_enhanced = (
    vegetation_mask &
    (rendvi < 0.25) &    # red-edge stress
    (mcari  < 0.15) &    # chlorophyll absorption dropping
    (ndre   < 0.45) &    # red-edge chlorophyll low
    (ndvi   > 0.40)      # but NDVI still looks healthy
)

symptomatic_enhanced = (
    vegetation_mask &
    (rendvi < 0.25) &
    (ndvi   < 0.40)      # now NDVI also dropping
)

print(f"\n  Pre-symptomatic (original):  "
      f"{(vegetation_mask & (rendvi < 0.25) & (ndvi > 0.40)).sum() * 100 / 10000:.2f} ha")
print(f"  Pre-symptomatic (enhanced):  "
      f"{pre_symptomatic_enhanced.sum() * 100 / 10000:.2f} ha")
print(f"  Symptomatic (enhanced):      "
      f"{symptomatic_enhanced.sum() * 100 / 10000:.2f} ha")

# ── stage 2: spread prediction ─────────────────────────────────────────────────
print("\nComputing spread risk...")
CONFIRMED_THRESHOLD = 70
seed_mask = (prob_enhanced >= CONFIRMED_THRESHOLD) & vegetation_mask

dist   = distance_transform_edt(~seed_mask).astype(float)
prox   = 1.0 - np.clip(dist / float(np.percentile(dist, 95)), 0, 1)
wbiased = shift(prox, shift=[8, 8], cval=0.0)
spread_raw = gaussian_filter(wbiased, sigma=25)
s_min, s_max = spread_raw.min(), spread_raw.max()
spread_risk = 100 * (spread_raw - s_min) / (s_max - s_min + 1e-9)
spread_risk = np.where(vegetation_mask & ~seed_mask, spread_risk, np.nan)

print(f"  Confirmed infection: {seed_mask.sum() * 100 / 10000:.2f} ha")
print(f"  Spread risk range:   {np.nanmin(spread_risk):.1f}% → {np.nanmax(spread_risk):.1f}%")

# ── index correlations ─────────────────────────────────────────────────────────
print("\n── Index correlations with infection probability ────────────")
for name, arr in [("NDVI", ndvi), ("RENDVI", rendvi), ("CIre", cire),
                   ("NDRE", ndre), ("EVI", evi), ("SAVI", savi),
                   ("MCARI", mcari), ("LAI", lai)]:
    valid = vegetation_mask & ~np.isnan(arr) & ~np.isnan(prob_enhanced)
    if valid.sum() > 100:
        r, p = pearsonr(arr[valid].flatten(), prob_enhanced[valid].flatten())
        flag = "✓" if abs(r) > 0.3 else " "
        print(f"  {flag} {name:<8} r = {r:+.3f}  p = {p:.4f}")

# ── validation: compare models ─────────────────────────────────────────────────
print("\n── Model comparison ─────────────────────────────────────────")
print(f"  Original 3-index mean score: {np.nanmean(prob_original):.1f}%")
print(f"  Enhanced 7-index mean score: {np.nanmean(prob_enhanced):.1f}%")

# load ground truth to compare
GEOJSON = "F1-coffee-leaf-rust-ethiopia/F1_CLR Survey Farm Level.geojson"
if os.path.exists(GEOJSON):
    import geopandas as gpd
    from scipy.stats import pearsonr as pr

    farms     = gpd.read_file(GEOJSON).to_crs("EPSG:32636")
    x_min     = float(x_coords.min())
    y_max_val = float(y_coords.max())
    x_res     = float(x_coords[1] - x_coords[0])
    y_res     = float(abs(y_coords[1] - y_coords[0]))

    orig_scores = []
    enh_scores  = []
    incidences  = []

    for _, farm in farms.iterrows():
        col = int((farm.geometry.x - x_min) / x_res)
        row = int((y_max_val - farm.geometry.y) / y_res)
        col = np.clip(col, 0, prob_enhanced.shape[1]-1)
        row = np.clip(row, 0, prob_enhanced.shape[0]-1)

        r0 = max(0, row-1); r1 = min(prob_enhanced.shape[0], row+2)
        c0 = max(0, col-1); c1 = min(prob_enhanced.shape[1], col+2)

        o = float(np.nanmean(prob_original[r0:r1, c0:c1]))
        e = float(np.nanmean(prob_enhanced[r0:r1, c0:c1]))

        if not np.isnan(o) and not np.isnan(e):
            orig_scores.append(o)
            enh_scores.append(e)
            incidences.append(float(farm["inc"]))

    if len(incidences) >= 5:
        r_orig, p_orig = pr(orig_scores, incidences)
        r_enh,  p_enh  = pr(enh_scores,  incidences)
        print(f"\n  vs confirmed field incidence (n={len(incidences)} farms):")
        print(f"  Original 3-index: r = {r_orig:.3f}  p = {p_orig:.4f}")
        print(f"  Enhanced 7-index: r = {r_enh:.3f}  p = {p_enh:.4f}")
        if r_enh > r_orig:
            print(f"  ✓ Enhanced model improves correlation by +{r_enh-r_orig:.3f}")
        else:
            print(f"  ~ No improvement in spectral correlation")

# ── plots ──────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)
cmap_spread = mcolors.LinearSegmentedColormap.from_list(
    "spread", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A"]
)
cmap_stage = mcolors.ListedColormap(["#5DCAA5", "#EF9F27", "#A32D2D"])

# ── MAP A — all 7 indices ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle(
    f"Crop health indices — {best_date}\n"
    "Seven spectral indices targeting chlorophyll breakdown from CLR infection",
    fontsize=14, fontweight="bold"
)

index_data = [
    ("NDVI",   ndvi,   "RdYlGn", 0.1,  0.85, "General vegetation health"),
    ("RENDVI", rendvi, "RdYlGn", 0.0,  0.5,  "Red-edge — primary CLR signal"),
    ("CIre",   cire,   "RdYlGn", 0.5,  3.0,  "Chlorophyll index red-edge"),
    ("NDRE",   ndre,   "RdYlGn", 0.3,  0.6,  "Red-edge chlorophyll content"),
    ("EVI",    evi,    "RdYlGn", 0.1,  0.7,  "Enhanced — dense canopy"),
    ("SAVI",   savi,   "RdYlGn", 0.1,  0.7,  "Soil-adjusted vegetation"),
    ("MCARI",  mcari,  "RdYlGn", 0.0,  0.8,  "Chlorophyll absorption ratio"),
]

for ax, (name, arr, cmap, vmin, vmax, desc) in zip(axes.flat, index_data):
    im = ax.imshow(np.where(vegetation_mask, arr, np.nan),
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{name}\n{desc}", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("map19_all_indices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map19_all_indices.png")

# ── MAP B — original vs enhanced probability ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(
    f"Original vs enhanced infection probability — {best_date}",
    fontsize=14, fontweight="bold"
)

im0 = axes[0].imshow(prob_original, cmap=cmap_risk, vmin=0, vmax=100)
axes[0].set_title("Original — 3 indices\n(NDVI + RENDVI + CIre)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="%")

im1 = axes[1].imshow(prob_enhanced, cmap=cmap_risk, vmin=0, vmax=100)
axes[1].set_title("Enhanced — 7 indices\n(+ NDRE + EVI + SAVI + MCARI)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("map20_original_vs_enhanced.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map20_original_vs_enhanced.png")

# ── MAP C — enhanced CLR stage map ────────────────────────────────────────────
clr_stage = np.where(vegetation_mask, 0, np.nan)
clr_stage = np.where(pre_symptomatic_enhanced, 1, clr_stage)
clr_stage = np.where(symptomatic_enhanced,     2, clr_stage)

bounds     = [-0.5, 0.5, 1.5, 2.5]
norm_stage = mcolors.BoundaryNorm(bounds, cmap_stage.N)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(
    f"Enhanced CLR detection — {best_date}\n"
    "MCARI + NDRE + RENDVI combination catches pre-symptomatic stress earlier",
    fontsize=13, fontweight="bold"
)

im0 = axes[0].imshow(clr_stage, cmap=cmap_stage, norm=norm_stage)
cbar = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
cbar.ax.set_yticklabels([
    "Healthy",
    f"Pre-symptomatic\n({pre_symptomatic_enhanced.sum() * 100 / 10000:.2f} ha)\n← invisible to naked eye",
    f"Symptomatic\n({symptomatic_enhanced.sum() * 100 / 10000:.2f} ha)"
])
axes[0].set_title("CLR detection stages\n(7-index enhanced model)")
axes[0].axis("off")

im1 = axes[1].imshow(spread_risk, cmap=cmap_spread, vmin=0, vmax=100)
axes[1].imshow(
    np.where(seed_mask, 1, np.nan),
    cmap=mcolors.ListedColormap(["#A32D2D"]),
    vmin=0, vmax=1
)
axes[1].set_title("Spread prediction\n(dark red = confirmed source)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Spread risk %")

plt.tight_layout()
plt.savefig("map21_enhanced_clr_stages.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map21_enhanced_clr_stages.png")

# ── save updated cache ─────────────────────────────────────────────────────────
cache = {}
try:
    with open(MODEL_CACHE, "rb") as f:
        import pickle
        cache = pickle.load(f)
except:
    pass

cache.update({
    "prob":                    prob_enhanced,
    "prob_original":           prob_original,
    "prob_enhanced":           prob_enhanced,
    "ndvi":                    ndvi,
    "rendvi":                  rendvi,
    "cire":                    cire,
    "ndre":                    ndre,
    "evi":                     evi,
    "savi":                    savi,
    "mcari":                   mcari,
    "lai":                     lai,
    "vegetation_mask":         vegetation_mask,
    "seed_mask":               seed_mask,
    "spread_risk":             spread_risk,
    "pre_symptomatic":         pre_symptomatic_enhanced,
    "symptomatic":             symptomatic_enhanced,
    "best_date":               best_date,
})

with open(MODEL_CACHE, "wb") as f:
    import pickle
    pickle.dump(cache, f)
print("Cache updated with enhanced indices")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n── Enhanced Model Summary ───────────────────────────────────")
print(f"  Date:                        {best_date}")
print(f"  Indices used:                NDVI, RENDVI, CIre, NDRE, EVI, SAVI, MCARI")
print(f"  Vegetation area:             {vegetation_mask.sum() * 100 / 10000:.0f} ha")
print(f"  Pre-symptomatic CLR:         {pre_symptomatic_enhanced.sum() * 100 / 10000:.2f} ha")
print(f"  Symptomatic CLR:             {symptomatic_enhanced.sum() * 100 / 10000:.2f} ha")
print(f"  Confirmed infection (>70%):  {seed_mask.sum() * 100 / 10000:.2f} ha")
print(f"  High spread risk (>70%):     "
      f"{((spread_risk >= 70) & vegetation_mask).sum() * 100 / 10000:.2f} ha")
print()
print("  MCARI specifically targets chlorophyll absorption —")
print("  drops in days 1-10 of CLR infection before NDVI changes")
print("  NDRE more sensitive to chlorophyll than NDVI —")
print("  detects stress 5-7 days earlier than NDVI alone")
print("────────────────────────────────────────────────────────────")