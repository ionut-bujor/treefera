import os
import zarr
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from scipy.stats import pearsonr, spearmanr

ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
GEOJSON   = "F1-coffee-leaf-rust-ethiopia/F1_CLR Survey Farm Level.geojson"

# ── load ground truth ──────────────────────────────────────────────────────────
print("Loading ground truth farm survey data...")
farms = gpd.read_file(GEOJSON)
print(f"  {len(farms)} farms, incidence {farms['inc'].min():.1f}–{farms['inc'].max():.1f}%")

# ── load zarr ─────────────────────────────────────────────────────────────────
print("\nLoading zarr...")
store     = zarr.open(ZARR_PATH, mode="r")
b04       = store["B04"][:]
b05       = store["B05"][:]
b07       = store["B07"][:]
b08       = store["B08"][:]
time_days = store["time"][:]
x_coords  = store["x"][:]
y_coords  = store["y"][:]
dates     = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")

# ── find time steps matching survey period Sep 2017 – Feb 2018 ────────────────
# this is CRITICAL — the field survey was done in this window
survey_dates = [
    pd.Timestamp("2017-09-01"),
    pd.Timestamp("2017-10-01"),
    pd.Timestamp("2017-11-01"),
    pd.Timestamp("2017-12-01"),
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2018-02-01"),
]

survey_indices = []
for sd in survey_dates:
    diffs = [abs((d - sd).days) for d in dates]
    idx   = int(np.argmin(diffs))
    survey_indices.append(idx)
    print(f"  Survey date {sd.date()} → zarr index {idx}: {dates[idx].date()}")

def norm_inverse(arr, lo, hi):
    return 1.0 - (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-9)

def compute_prob(t):
    red  = b04[t].astype(float) / 10000
    re1  = b05[t].astype(float) / 10000
    re3  = b07[t].astype(float) / 10000
    nir  = b08[t].astype(float) / 10000
    mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_t   = np.where(mask, (nir - red) / (nir + red + 1e-9), np.nan)
        rendvi_t = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)
        cire_t   = np.where(mask, (re3 / re1) - 1,                   np.nan)
    veg = mask & (ndvi_t > 0.2) & (ndvi_t < 0.85)
    p = (
        0.25 * norm_inverse(ndvi_t,   0.2, 0.8) +
        0.40 * norm_inverse(rendvi_t, 0.0, 0.4) +
        0.35 * norm_inverse(cire_t,   0.5, 3.0)
    ) * 100
    return np.where(veg, np.clip(p, 0, 100), np.nan), ndvi_t, rendvi_t, veg

# ── compute mean probability across survey period ──────────────────────────────
print("\nComputing probability maps for survey period...")
prob_stack = []
for t in survey_indices:
    p, _, _, _ = compute_prob(t)
    prob_stack.append(p)
    print(f"  {dates[t].date()} — mean score: {np.nanmean(p):.1f}%")

prob_survey = np.nanmean(np.stack(prob_stack), axis=0)
print(f"\nMean probability across survey period: {np.nanmean(prob_survey):.1f}%")

# also compute for Jan 2018 specifically — peak CLR month
_, _, _, veg_mask = compute_prob(survey_indices[-1])

# ── reproject farms to UTM ────────────────────────────────────────────────────
print("\nExtracting model scores at farm locations...")
farms_utm = farms.to_crs("EPSG:32636")

x_min = float(x_coords.min())
x_max = float(x_coords.max())
y_max_coord = float(y_coords.max())
x_res = float(x_coords[1] - x_coords[0])
y_res = float(abs(y_coords[1] - y_coords[0]))

results = []
for _, farm in farms_utm.iterrows():
    fx = farm.geometry.x
    fy = farm.geometry.y

    in_bounds = (x_min <= fx <= x_max and
                 float(y_coords.min()) <= fy <= float(y_coords.max()))

    if not in_bounds:
        print(f"  Farm {farm['farm']} outside bounds")
        results.append({
            "farm": farm["farm"], "inc": farm["inc"], "sev": farm["sev"],
            "altitude": farm["altitude"], "model_score": np.nan,
            "fx": fx, "fy": fy, "in_bounds": False
        })
        continue

    col = int((fx - x_min) / x_res)
    row = int((y_max_coord - fy) / y_res)
    col = np.clip(col, 0, prob_survey.shape[1] - 1)
    row = np.clip(row, 0, prob_survey.shape[0] - 1)

    # sample 3x3 neighbourhood to account for GPS uncertainty
    r0 = max(0, row - 1)
    r1 = min(prob_survey.shape[0], row + 2)
    c0 = max(0, col - 1)
    c1 = min(prob_survey.shape[1], col + 2)
    patch = prob_survey[r0:r1, c0:c1]
    score = float(np.nanmean(patch))

    results.append({
        "farm": farm["farm"], "inc": farm["inc"], "sev": farm["sev"],
        "altitude": farm["altitude"], "model_score": score,
        "fx": fx, "fy": fy, "in_bounds": True
    })

results_df = pd.DataFrame(results)
valid = results_df[results_df["in_bounds"] & ~results_df["model_score"].isna()].copy()
print(f"  {len(valid)} of {len(results_df)} farms within study area")

# ── correlations ───────────────────────────────────────────────────────────────
print("\n── Ground Truth Validation ─────────────────────────────────")
print(f"  Using survey-period imagery (Sep 2017 – Feb 2018)")
print(f"  Mean probability across survey period")

if len(valid) >= 5:
    r_inc, p_inc = pearsonr(valid["model_score"], valid["inc"])
    r_sev, p_sev = pearsonr(valid["model_score"], valid["sev"])
    r_inc_sp, _  = spearmanr(valid["model_score"], valid["inc"])
    r_sev_sp, _  = spearmanr(valid["model_score"], valid["sev"])
    r_alt, p_alt = pearsonr(valid["altitude"], valid["inc"])

    print(f"\n  Farms used:      {len(valid)}")
    print(f"  Model score:     {valid['model_score'].min():.1f}% → {valid['model_score'].max():.1f}%")
    print(f"  Field incidence: {valid['inc'].min():.1f}% → {valid['inc'].max():.1f}%")
    print()
    print(f"  Model score vs field INCIDENCE:")
    print(f"    Pearson  r = {r_inc:.3f}  p = {p_inc:.4f}")
    print(f"    Spearman r = {r_inc_sp:.3f}")
    if r_inc > 0.3 and p_inc < 0.05:
        print(f"    ✓ Significant positive correlation with confirmed CLR incidence")
    elif r_inc > 0.2:
        print(f"    ~ Positive trend (small n={len(valid)} limits significance)")
    elif r_inc < -0.2:
        print(f"    ✗ Negative correlation — likely temporal or spatial mismatch")
    print()
    print(f"  Model score vs field SEVERITY:")
    print(f"    Pearson  r = {r_sev:.3f}  p = {p_sev:.4f}")
    print(f"    Spearman r = {r_sev_sp:.3f}")
    print()
    print(f"  Altitude vs incidence (literature sanity check):")
    print(f"    r = {r_alt:.3f}  p = {p_alt:.4f}")
    if r_alt < -0.3:
        print(f"    ✓ Lower altitude = higher incidence — confirms Belachew et al. 2020")
    print()

    # ── farm table ─────────────────────────────────────────────────────────────
    print(f"  {'Farm':<6} {'Altitude':>8} {'Inc (field)':>12} {'Sev (field)':>12} {'Model score':>12}")
    print("  " + "-" * 54)
    for _, row in valid.sort_values("inc", ascending=False).iterrows():
        print(f"  {int(row['farm']):<6} {row['altitude']:>7.0f}m "
              f"{row['inc']:>11.1f}% {row['sev']:>11.1f}% "
              f"{row['model_score']:>11.1f}%")
    print("────────────────────────────────────────────────────────────")

    # ── plots ──────────────────────────────────────────────────────────────────
    cmap_risk = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Ground truth validation — model score vs confirmed CLR field survey\n"
        "Belachew et al. (2020) — Sep 2017 to Feb 2018, SW Ethiopia",
        fontsize=13, fontweight="bold"
    )

    # scatter: model vs incidence
    ax = axes[0]
    sc = ax.scatter(valid["inc"], valid["model_score"],
                    c=valid["altitude"], cmap="RdYlGn_r",
                    s=100, zorder=5, edgecolors="white", linewidths=0.8)
    if len(valid) >= 3:
        m, b = np.polyfit(valid["inc"], valid["model_score"], 1)
        xl = np.linspace(valid["inc"].min(), valid["inc"].max(), 100)
        ax.plot(xl, m * xl + b, color="#A32D2D", linewidth=2)
    for _, row in valid.iterrows():
        ax.annotate(f"F{int(row['farm'])}",
                    (row["inc"], row["model_score"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.colorbar(sc, ax=ax, label="Altitude (m)")
    ax.set_xlabel("Field incidence (%) — confirmed CLR", fontsize=11)
    ax.set_ylabel("Model probability score (%)", fontsize=11)
    ax.set_title(f"Model vs confirmed incidence\nr = {r_inc:.3f}  p = {p_inc:.4f}")

    # scatter: model vs severity
    ax = axes[1]
    ax.scatter(valid["sev"], valid["model_score"],
               c=valid["altitude"], cmap="RdYlGn_r",
               s=100, zorder=5, edgecolors="white", linewidths=0.8)
    if len(valid) >= 3:
        m2, b2 = np.polyfit(valid["sev"], valid["model_score"], 1)
        xl2 = np.linspace(valid["sev"].min(), valid["sev"].max(), 100)
        axes[1].plot(xl2, m2 * xl2 + b2, color="#A32D2D", linewidth=2)
    ax.set_xlabel("Field severity (%) — confirmed CLR", fontsize=11)
    ax.set_ylabel("Model probability score (%)", fontsize=11)
    ax.set_title(f"Model vs confirmed severity\nr = {r_sev:.3f}  p = {p_sev:.4f}")

    # map with farm points
    ax = axes[2]
    im = ax.imshow(prob_survey, cmap=cmap_risk, vmin=0, vmax=100,
                   extent=[x_coords.min(), x_coords.max(),
                           y_coords.min(), y_coords.max()],
                   aspect="auto", origin="upper")
    valid_utm = valid.copy()
    sc2 = ax.scatter(valid["fx"], valid["fy"],
                     c=valid["inc"], cmap="RdYlGn_r",
                     s=120, zorder=5, edgecolors="white", linewidths=1.5,
                     vmin=valid["inc"].min(), vmax=valid["inc"].max())
    for _, row in valid.iterrows():
        ax.annotate(f"F{int(row['farm'])}\n{row['inc']:.0f}%",
                    (row["fx"], row["fy"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", alpha=0.6))
    plt.colorbar(im,  ax=ax, fraction=0.046, pad=0.04, label="Model score %")
    ax.set_title("Farm survey points on model map\n(label = field incidence %)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("map14_ground_truth_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: map14_ground_truth_validation.png")

    valid[["farm", "altitude", "inc", "sev", "model_score"]].to_csv(
        "ground_truth_results.csv", index=False
    )
    print("Saved: ground_truth_results.csv")

else:
    print(f"\n  Only {len(valid)} farms in bounds — check coordinates")
    print(f"  Raster bounds (UTM): x={x_min:.0f}–{x_max:.0f}")
    print(f"  Farm UTM coords:")
    for _, r in farms_utm.iterrows():
        print(f"    Farm {r['farm']}: {r.geometry.x:.0f}, {r.geometry.y:.0f}")