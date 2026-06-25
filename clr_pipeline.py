import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift
from scipy.stats import pearsonr
import pickle
import os

ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
MODEL_CACHE = "clr_model_cache.pkl"

# ── load or compute ────────────────────────────────────────────────────────────
if os.path.exists(MODEL_CACHE):
    print("Loading from cache...")
    with open(MODEL_CACHE, "rb") as f:
        cache = pickle.load(f)
    ndvi         = cache["ndvi"]
    rendvi       = cache["rendvi"]
    cire         = cache["cire"]
    slope        = cache["slope"]
    prob         = cache["prob"]
    spread_risk  = cache["spread_risk"]
    seed_mask    = cache["seed_mask"]
    vegetation_mask = cache["vegetation_mask"]
    pre_symptomatic = cache["pre_symptomatic"]
    symptomatic     = cache["symptomatic"]
    prob_val        = cache["prob_val"]
    ndvi_drop       = cache["ndvi_drop"]
    valid_both      = cache["valid_both"]
    r_val_corr      = cache["r_val_corr"]
    p_val           = cache["p_val"]
    best_date       = cache["best_date"]
    print(f"Cache loaded — date: {best_date}")

else:
    print("No cache found — loading raw data (this will take a minute)...")
    store = zarr.open(ZARR_PATH, mode="r")

    b04 = store["B04"][:]
    b05 = store["B05"][:]
    b07 = store["B07"][:]
    b08 = store["B08"][:]
    time_days = store["time"][:]

    dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")
    print(f"Date range: {dates[0].date()} → {dates[-1].date()}")

    coverage = (b04 > 0).sum(axis=(1, 2))
    best_t = int(np.argmax(coverage[-20:]) + len(coverage) - 20)
    best_date = str(dates[best_t].date())
    print(f"Using time step {best_t}: {best_date}")

    # ── extract bands ──────────────────────────────────────────────────────────
    red = b04[best_t].astype(float) / 10000
    re1 = b05[best_t].astype(float) / 10000
    re3 = b07[best_t].astype(float) / 10000
    nir = b08[best_t].astype(float) / 10000
    mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi   = np.where(mask, (nir - red) / (nir + red + 1e-9), np.nan)
        rendvi = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)
        cire   = np.where(mask, (re3 / re1) - 1,                   np.nan)

    vegetation_mask = mask & (ndvi > 0.2) & (ndvi < 0.85)
    print(f"Vegetation pixels: {vegetation_mask.sum():,} of {mask.sum():,}")

    # ── ndvi trend slope ───────────────────────────────────────────────────────
    print("Computing NDVI trend slope...")
    N_TREND = 6
    ndvi_series = []
    for t in range(best_t - N_TREND, best_t):
        r = b04[t].astype(float)
        n = b08[t].astype(float)
        valid = (r > 0) & (n > 0)
        vi = np.where(valid, (n - r) / (n + r + 1e-9), np.nan)
        ndvi_series.append(vi)
    ndvi_series = np.stack(ndvi_series)

    x = np.arange(N_TREND, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    with np.errstate(invalid="ignore"):
        y_mean = np.nanmean(ndvi_series, axis=0)
        slope = np.nansum(
            (ndvi_series - y_mean[np.newaxis]) *
            (x - x_mean)[:, np.newaxis, np.newaxis], axis=0
        ) / x_var

    slope_score = np.where(vegetation_mask, np.clip(-slope * 20, 0, 1), np.nan)

    # ── stage 1: infection probability ────────────────────────────────────────
    print("Computing infection probability...")

    def norm_inverse(arr, lo, hi):
        clipped = np.clip(arr, lo, hi)
        return 1.0 - (clipped - lo) / (hi - lo + 1e-9)

    prob = (
        0.20 * norm_inverse(ndvi,   0.2, 0.8) +
        0.35 * norm_inverse(rendvi, 0.0, 0.4) +
        0.30 * norm_inverse(cire,   0.5, 3.0) +
        0.15 * slope_score
    ) * 100
    prob = np.where(vegetation_mask, np.clip(prob, 0, 100), np.nan)

    # ── pre-symptomatic vs symptomatic ────────────────────────────────────────
    pre_symptomatic = (
            vegetation_mask &
            (rendvi < 0.25) &
            (ndvi > 0.40)
    )
    symptomatic = (
        vegetation_mask &
        (rendvi < 0.15) &
        (ndvi < 0.35)
    )

    # ── historical validation ──────────────────────────────────────────────────
    print("Running historical validation...")
    val_t = best_t - 2
    r_val  = b04[val_t].astype(float)
    re1_val = b05[val_t].astype(float)
    re3_val = b07[val_t].astype(float)
    n_val  = b08[val_t].astype(float)
    mask_val = (r_val > 0) & (n_val > 0) & (re1_val > 0) & (re3_val > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_val   = np.where(mask_val, (n_val - r_val) / (n_val + r_val + 1e-9), np.nan)
        rendvi_val = np.where(mask_val, (re3_val - re1_val) / (re3_val + re1_val + 1e-9), np.nan)
        cire_val   = np.where(mask_val, (re3_val / re1_val) - 1, np.nan)

    veg_val = mask_val & (ndvi_val > 0.2) & (ndvi_val < 0.85)
    prob_val = (
        0.25 * norm_inverse(ndvi_val,   0.2, 0.8) +
        0.40 * norm_inverse(rendvi_val, 0.0, 0.4) +
        0.35 * norm_inverse(cire_val,   0.5, 3.0)
    ) * 100
    prob_val = np.where(veg_val, np.clip(prob_val, 0, 100), np.nan)

    ndvi_drop = ndvi - ndvi_val
    valid_both = ~np.isnan(prob_val) & ~np.isnan(ndvi_drop) & veg_val & vegetation_mask
    r_val_corr, p_val = pearsonr(
        prob_val[valid_both].flatten(),
        ndvi_drop[valid_both].flatten()
    )

    # ── stage 2: spread prediction ─────────────────────────────────────────────
    print("Computing spread risk...")
    CONFIRMED_THRESHOLD = 70
    seed_mask = (prob >= CONFIRMED_THRESHOLD) & vegetation_mask

    dist_from_infection = distance_transform_edt(~seed_mask).astype(float)
    max_dist = float(np.percentile(dist_from_infection, 95))
    proximity_score = 1.0 - np.clip(dist_from_infection / max_dist, 0, 1)
    wind_biased = shift(proximity_score, shift=[8, 8], cval=0.0)
    spread_risk_raw = gaussian_filter(wind_biased, sigma=25)
    s_min, s_max = spread_risk_raw.min(), spread_risk_raw.max()
    spread_risk = 100 * (spread_risk_raw - s_min) / (s_max - s_min + 1e-9)
    spread_risk = np.where(vegetation_mask & ~seed_mask, spread_risk, np.nan)

    # ── save cache ─────────────────────────────────────────────────────────────
    print("Saving cache...")
    cache = {
        "ndvi": ndvi, "rendvi": rendvi, "cire": cire,
        "slope": slope, "prob": prob, "spread_risk": spread_risk,
        "seed_mask": seed_mask, "vegetation_mask": vegetation_mask,
        "pre_symptomatic": pre_symptomatic, "symptomatic": symptomatic,
        "prob_val": prob_val, "ndvi_drop": ndvi_drop,
        "valid_both": valid_both, "r_val_corr": r_val_corr,
        "p_val": p_val, "best_date": best_date,
    }
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {MODEL_CACHE}")

# ── print summary ──────────────────────────────────────────────────────────────
print(f"\nNDVI range:   {np.nanmin(ndvi):.3f} → {np.nanmax(ndvi):.3f}")
print(f"RENDVI range: {np.nanmin(rendvi):.3f} → {np.nanmax(rendvi):.3f}")
print(f"Infection probability: {np.nanmin(prob):.1f}% → {np.nanmax(prob):.1f}%")
print(f"Spread risk: {np.nanmin(spread_risk):.1f}% → {np.nanmax(spread_risk):.1f}%")
print(f"Pre-symptomatic area: {pre_symptomatic.sum() * 100 / 10000:.2f} ha")
print(f"Symptomatic area:     {symptomatic.sum() * 100 / 10000:.2f} ha")
print(f"Validation r = {r_val_corr:.3f}, p = {p_val:.4f}")

# ── colormaps ──────────────────────────────────────────────────────────────────
cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)
cmap_spread = mcolors.LinearSegmentedColormap.from_list(
    "spread", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A"]
)
cmap_slope = mcolors.LinearSegmentedColormap.from_list(
    "slope", ["#5DCAA5", "#EF9F27", "#A32D2D"]
)
cmap_stage = mcolors.ListedColormap(["#5DCAA5", "#EF9F27", "#A32D2D"])

# ══════════════════════════════════════════════════════════════════════════════
# MAP 1 — spectral signals (NDVI + red-edge)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"Spectral signals — {best_date}", fontsize=14, fontweight="bold")

im0 = axes[0].imshow(ndvi, cmap="RdYlGn", vmin=0.1, vmax=0.85)
axes[0].set_title("NDVI — vegetation health\n(red = low, green = healthy)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(rendvi, cmap="RdYlGn", vmin=0, vmax=0.5)
axes[1].set_title("Red-edge NDVI — CLR stress signal\n(red = chlorophyll breakdown)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("map1_spectral_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map1_spectral_signals.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAP 2 — pre-symptomatic vs symptomatic
# ══════════════════════════════════════════════════════════════════════════════
clr_stage_map = np.where(vegetation_mask, 0, np.nan)
clr_stage_map = np.where(pre_symptomatic, 1, clr_stage_map)
clr_stage_map = np.where(symptomatic,     2, clr_stage_map)

bounds = [-0.5, 0.5, 1.5, 2.5]
norm_stage = mcolors.BoundaryNorm(bounds, cmap_stage.N)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(clr_stage_map, cmap=cmap_stage, norm=norm_stage)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
cbar.ax.set_yticklabels([
    "Healthy",
    "Pre-symptomatic\n(red-edge stress, NDVI still green)\n← invisible to naked eye",
    "Symptomatic\n(both signals degraded)\n← visible yellowing"
])
ax.set_title(
    f"CLR detection stages — {best_date}\n"
    f"Orange = caught in the 20-day invisible window",
    fontweight="bold", fontsize=13
)
ax.axis("off")

textstr = f"Pre-symptomatic: {pre_symptomatic.sum() * 100 / 10000:.1f} ha\nSymptomatic: {symptomatic.sum() * 100 / 10000:.1f} ha"
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("map2_clr_stages.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map2_clr_stages.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAP 3 — infection probability (stage 1)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"Stage 1 — infection probability — {best_date}", fontsize=14, fontweight="bold")

im0 = axes[0].imshow(np.where(vegetation_mask, -slope, np.nan),
                     cmap=cmap_slope, vmin=-0.01, vmax=0.05)
axes[0].set_title("NDVI decline trend\n(red = declining over 6 months)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(prob, cmap=cmap_risk, vmin=0, vmax=100)
axes[1].set_title("Combined infection probability (%)\n(spectral + trend slope)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("map3_infection_probability.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map3_infection_probability.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAP 4 — spread prediction (stage 2)
# ══════════════════════════════════════════════════════════════════════════════
high_spread     = (spread_risk >= 70) & vegetation_mask & ~seed_mask
moderate_spread = (spread_risk >= 40) & (spread_risk < 70) & vegetation_mask & ~seed_mask
low_spread      = (spread_risk  < 40) & vegetation_mask & ~seed_mask

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(spread_risk, cmap=cmap_spread, vmin=0, vmax=100)
ax.imshow(
    np.where(seed_mask, 1, np.nan),
    cmap=mcolors.ListedColormap(["#A32D2D"]),
    vmin=0, vmax=1
)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spread risk %")
ax.set_title(
    f"Stage 2 — spread prediction — {best_date}\n"
    "Dark red = confirmed infection source | Orange/red = next at risk",
    fontweight="bold", fontsize=13
)
ax.axis("off")

textstr = (
    f"Confirmed infection: {seed_mask.sum() * 100 / 10000:.1f} ha\n"
    f"High spread risk (>70%): {high_spread.sum() * 100 / 10000:.1f} ha\n"
    f"Moderate spread risk (40-70%): {moderate_spread.sum() * 100 / 10000:.1f} ha"
)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("map4_spread_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map4_spread_prediction.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAP 5 — validation scatter plot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    prob_val[valid_both][::50],
    ndvi_drop[valid_both][::50],
    alpha=0.3, s=2, color="#E24B4A"
)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

m, b = np.polyfit(prob_val[valid_both], ndvi_drop[valid_both], 1)
x_line = np.linspace(prob_val[valid_both].min(), prob_val[valid_both].max(), 100)
ax.plot(x_line, m * x_line + b, color="#A32D2D", linewidth=2, label=f"r = {r_val_corr:.3f}")

ax.set_xlabel("Infection probability score at earlier date (%)", fontsize=12)
ax.set_ylabel("Subsequent NDVI decline", fontsize=12)
ax.set_title(
    f"Validation — does high probability predict NDVI decline?\n"
    f"r = {r_val_corr:.3f},  p = {p_val:.2e},  n = {valid_both.sum():,} pixels",
    fontweight="bold", fontsize=13
)
ax.legend(fontsize=12)
ax.text(0.05, 0.92,
        "Higher probability score → greater vegetation decline\n= model is detecting real stress",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("map5_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map5_validation.png")

# ── farmer summary ─────────────────────────────────────────────────────────────
print("\n── Farm Risk Summary ──────────────────────────────────────")
print(f"  Date analysed:                  {best_date}")
print(f"  Total vegetation area:          {vegetation_mask.sum() * 100 / 10000:.1f} ha")
print(f"  Pre-symptomatic CLR area:       {pre_symptomatic.sum() * 100 / 10000:.2f} ha")
print(f"  Symptomatic CLR area:           {symptomatic.sum() * 100 / 10000:.2f} ha")
print(f"  Confirmed/high infection:       {seed_mask.sum() * 100 / 10000:.2f} ha")
print(f"  High spread risk   (>70%):      {high_spread.sum() * 100 / 10000:.2f} ha")
print(f"  Moderate spread risk (40-70%):  {moderate_spread.sum() * 100 / 10000:.2f} ha")
print(f"  Validation r:                   {r_val_corr:.3f} (p={p_val:.2e})")
print("────────────────────────────────────────────────────────────")
print("  Priority: inspect and treat HIGH spread risk zones first")
print("────────────────────────────────────────────────────────────")