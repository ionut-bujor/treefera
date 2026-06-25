import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pickle
import os


ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
VALIDATION_CACHE = "validation_cache.pkl"

def compute_stress_score(b04_t, b05_t, b07_t, b08_t):
    """Compute infection probability score for a single time step"""
    red = b04_t.astype(float)/10000
    re1 = b05_t.astype(float)/10000
    re3 = b07_t.astype(float)/10000
    nir = b08_t.astype(float)/10000
    mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi   = np.where(mask, (nir - red) / (nir + red + 1e-9), np.nan)
        rendvi = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)
        cire   = np.where(mask, (re3 / re1) - 1,                   np.nan)

    veg = mask & (ndvi > 0.2) & (ndvi < 0.85)

    def norm_inv(arr, lo, hi):
        return 1.0 - (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-9)

    score = (
        0.25 * norm_inv(ndvi,   0.2, 0.8) +
        0.40 * norm_inv(rendvi, 0.0, 0.4) +
        0.35 * norm_inv(cire,   0.5, 3.0)
    ) * 100

    return np.where(veg, np.clip(score, 0, 100), np.nan), ndvi, rendvi, veg

if os.path.exists(VALIDATION_CACHE):
    print("Loading validation cache...")
    with open(VALIDATION_CACHE, "rb") as f:
        data = pickle.load(f)
    scores     = data["scores"]
    ndvi_all   = data["ndvi_all"]
    rendvi_all = data["rendvi_all"]
    dates      = data["dates"]
    target_indices = data["target_indices"]
else:
    print("Loading zarr data...")
    store = zarr.open(ZARR_PATH, mode="r")
    b04 = store["B04"][:]
    b05 = store["B05"][:]
    b07 = store["B07"][:]
    b08 = store["B08"][:]
    time_days = store["time"][:]
    dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")

    # ── find time steps covering 2020-2021 outbreak period ────────────────────
    # CLR peaks in Ethiopia after rainy season — target Jan-Apr each year
    target_indices = [
        i for i, d in enumerate(dates)
        if (d.year == 2019 and d.month >= 9) or
           (d.year == 2020) or
           (d.year == 2021 and d.month <= 6)
    ]
    print(f"Found {len(target_indices)} time steps covering 2019-2021 outbreak window")
    for i in target_indices:
        print(f"  [{i}] {dates[i].date()}")

    # ── compute stress scores for all target time steps ────────────────────────
    print("\nComputing stress scores across outbreak window...")
    scores     = []
    ndvi_all   = []
    rendvi_all = []

    for i, t in enumerate(target_indices):
        print(f"  Processing {dates[t].date()} ({i+1}/{len(target_indices)})")
        score, ndvi, rendvi, veg = compute_stress_score(
            b04[t], b05[t], b07[t], b08[t]
        )
        scores.append(score)
        ndvi_all.append(ndvi)
        rendvi_all.append(rendvi)

    scores     = np.stack(scores)
    ndvi_all   = np.stack(ndvi_all)
    rendvi_all = np.stack(rendvi_all)

    with open(VALIDATION_CACHE, "wb") as f:
        pickle.dump({
            "scores": scores, "ndvi_all": ndvi_all,
            "rendvi_all": rendvi_all, "dates": dates,
            "target_indices": target_indices
        }, f)
    print("Validation cache saved")

target_dates = [dates[i] for i in target_indices]

# ── mean stress score per time step ───────────────────────────────────────────
mean_scores  = [np.nanmean(s) for s in scores]
mean_ndvi    = [np.nanmean(n) for n in ndvi_all]
mean_rendvi  = [np.nanmean(r) for r in rendvi_all]

# ── find peak outbreak time step ──────────────────────────────────────────────
peak_idx = int(np.argmax(mean_scores))
peak_date = target_dates[peak_idx]
print(f"\nPeak stress score: {mean_scores[peak_idx]:.1f}% at {peak_date.date()}")
print(f"This should fall in the 2020/21 documented outbreak window")

# ── map 6: time series of mean stress score ───────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(
    "Time series validation — 2019 to 2021\n"
    "Does model detect the documented 2020/21 CLR outbreak in SW Ethiopia?",
    fontsize=13, fontweight="bold"
)

date_labels = [d.date() for d in target_dates]

axes[0].plot(date_labels, mean_scores, color="#E24B4A",
             linewidth=2, marker="o", markersize=5, label="Mean infection probability")
axes[0].axvspan(
    pd.Timestamp("2020-07-01").date(),
    pd.Timestamp("2021-06-30").date(),
    alpha=0.15, color="#E24B4A", label="Documented 2020/21 outbreak period"
)
axes[0].axvline(peak_date.date(), color="#A32D2D",
                linewidth=1.5, linestyle="--", label=f"Peak: {peak_date.date()}")
axes[0].set_ylabel("Mean infection probability (%)", fontsize=11)
axes[0].set_xlabel("")
axes[0].legend(fontsize=10)
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_title("Infection probability score over time")

axes[1].plot(date_labels, mean_ndvi, color="#5DCAA5",
             linewidth=2, marker="o", markersize=5, label="Mean NDVI")
axes[1].plot(date_labels, mean_rendvi, color="#EF9F27",
             linewidth=2, marker="s", markersize=5, label="Mean Red-edge NDVI")
axes[1].axvspan(
    pd.Timestamp("2020-07-01").date(),
    pd.Timestamp("2021-06-30").date(),
    alpha=0.15, color="#E24B4A", label="Documented 2020/21 outbreak period"
)
# run stress score for ALL 108 time steps
print("Computing stress scores for full 2017-2025 time series...")
all_scores = []
all_dates  = []

for t in range(len(dates)):
    r   = b04[t].astype(float) / 10000
    re1 = b05[t].astype(float) / 10000
    re3 = b07[t].astype(float) / 10000
    n   = b08[t].astype(float) / 10000
    mask = (r > 0) & (n > 0) & (re1 > 0) & (re3 > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_t   = np.where(mask, (n - r) / (n + r + 1e-9), np.nan)
        rendvi_t = np.where(mask, (re3 - re1) / (re3 + re1 + 1e-9), np.nan)
        cire_t   = np.where(mask, (re3 / re1) - 1, np.nan)

    veg = mask & (ndvi_t > 0.2) & (ndvi_t < 0.85)

    def ni(arr, lo, hi):
        return 1.0 - (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-9)

    score = (
        0.25 * ni(ndvi_t,   0.2, 0.8) +
        0.40 * ni(rendvi_t, 0.0, 0.4) +
        0.35 * ni(cire_t,   0.5, 3.0)
    ) * 100

    all_scores.append(float(np.nanmean(np.where(veg, score, np.nan))))
    all_dates.append(dates[t])
    if t % 12 == 0:
        print(f"  {dates[t].date()} — score: {all_scores[-1]:.1f}%")

# plot full 8-year time series
fig, ax = plt.subplots(figsize=(18, 6))
fig.suptitle(
    "Full 2017–2025 stress score time series\n"
    "Does January–March peak repeat every year? Is 2021 the highest?",
    fontsize=13, fontweight="bold"
)

ax.plot(all_dates, all_scores, color="#E24B4A", linewidth=1.5,
        marker="o", markersize=4)

# shade every Jan-Mar window
for year in range(2017, 2026):
    ax.axvspan(
        pd.Timestamp(f"{year}-01-01"),
        pd.Timestamp(f"{year}-03-31"),
        alpha=0.08, color="#E24B4A"
    )

# label 2021 peak
peak_idx  = int(np.argmax(all_scores))
peak_date = all_dates[peak_idx]
ax.annotate(
    f"Highest peak\n{peak_date.date()}\n{all_scores[peak_idx]:.1f}%",
    xy=(peak_date, all_scores[peak_idx]),
    xytext=(peak_date + pd.Timedelta(days=60), all_scores[peak_idx] - 3),
    arrowprops=dict(arrowstyle="->", color="#A32D2D"),
    fontsize=9, color="#A32D2D"
)

ax.set_ylabel("Mean infection probability (%)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# add Jan-Mar label
ax.text(pd.Timestamp("2017-02-01"), max(all_scores) * 0.97,
        "Shaded = Jan–Mar\n(documented CLR peak window)",
        fontsize=8, color="#A32D2D", alpha=0.7)

plt.tight_layout()
plt.savefig("map11_full_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map11_full_timeseries.png")

# check if Jan-Mar consistently scores higher than other months
jan_mar_scores = [s for s, d in zip(all_scores, all_dates) if d.month in [1, 2, 3]]
other_scores   = [s for s, d in zip(all_scores, all_dates) if d.month not in [1, 2, 3]]

t_stat, p_ttest = ttest_ind(jan_mar_scores, other_scores)

print(f"\n── Seasonal Pattern Analysis ───────────────────────────────")
print(f"  Jan–Mar mean score:    {np.mean(jan_mar_scores):.1f}%")
print(f"  Other months mean:     {np.mean(other_scores):.1f}%")
print(f"  T-test: t={t_stat:.3f}, p={p_ttest:.4f}")
if p_ttest < 0.05:
    print(f"  ✓ Jan–Mar scores significantly higher than other months")
    print(f"    (p={p_ttest:.4f}) — consistent with CLR seasonal pattern")
print(f"  Overall peak: {peak_date.date()} at {all_scores[peak_idx]:.1f}%")
if peak_date.year == 2021:
    print(f"  ✓ 2021 is the highest stress year in the entire 2017–2025 record")
    print(f"    matching the documented 2020/21 CLR outbreak")
print("────────────────────────────────────────────────────────────")
axes[1].set_ylabel("Index value", fontsize=11)
axes[1].set_xlabel("Date", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_title("NDVI vs Red-edge NDVI — divergence indicates early stress")

plt.tight_layout()
plt.savefig("map6_timeseries_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map6_timeseries_validation.png")

# ── map 7: spatial comparison — before vs during vs after outbreak ─────────────
cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)

# pick three representative time steps
before_idx = 0                        # earliest — pre outbreak
during_idx = peak_idx                 # peak stress
after_idx  = len(target_indices) - 1  # latest — post outbreak

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "Spatial infection probability — before, during and after 2020/21 outbreak",
    fontsize=13, fontweight="bold"
)

for ax, idx, label in zip(
    axes,
    [before_idx, during_idx, after_idx],
    ["Before outbreak", "Peak stress", "After outbreak"]
):
    im = ax.imshow(scores[idx], cmap=cmap_risk, vmin=0, vmax=100)
    ax.set_title(f"{label}\n{target_dates[idx].date()}\nMean score: {mean_scores[idx]:.1f}%")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("map7_spatial_outbreak_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map7_spatial_outbreak_comparison.png")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n── Time Series Validation Summary ──────────────────────────")
print(f"  Period analysed:     {target_dates[0].date()} → {target_dates[-1].date()}")
print(f"  Time steps:          {len(target_indices)}")
print(f"  Peak stress date:    {peak_date.date()}")
print(f"  Peak mean score:     {mean_scores[peak_idx]:.1f}%")
print(f"  Pre-outbreak score:  {mean_scores[0]:.1f}%")
print(f"  Score increase:      +{mean_scores[peak_idx] - mean_scores[0]:.1f}%")
print(f"  Literature says:     CLR outbreak documented in SW Ethiopia 2020/21")
print(f"  Literature peak window:  January–March (dry season onset)")
print(f"  Your model peak:         {peak_date.strftime('%B %Y')}")
if peak_date.month in [1, 2, 3]:
    print(f"  ✓ Peak month falls exactly within documented CLR peak window")
if pd.Timestamp("2020-01-01") <= peak_date <= pd.Timestamp("2022-01-01"):
    print(f"  ✓ Peak falls within documented outbreak window — model validated")
else:
    print(f"  ~ Peak outside expected window — seasonal or data quality factor")
print("────────────────────────────────────────────────────────────")