# ── Stage 2: spread prediction ─────────────────────────────────────────────────
# Takes the probability map from Stage 1 and predicts which zones are next

from scipy.ndimage import distance_transform_edt, gaussian_filter

# ── identify confirmed / high infection seed zones ────────────────────────────
CONFIRMED_THRESHOLD = 70   # zones above this % treated as infection sources

seed_mask = (prob >= CONFIRMED_THRESHOLD) & mask

print(f"Confirmed/high infection pixels: {seed_mask.sum():,}")
print(f"As % of valid area: {100 * seed_mask.sum() / mask.sum():.1f}%")

# ── distance from nearest infection source ────────────────────────────────────
# distance_transform_edt gives distance (in pixels) from nearest seed
# invert so: close to infection = high risk
dist_from_infection = distance_transform_edt(~seed_mask)
dist_from_infection = np.where(mask, dist_from_infection, np.nan)

# normalise: 0 = far away (low spread risk), 1 = adjacent (high spread risk)
max_dist = np.nanpercentile(dist_from_infection, 95)
proximity_score = np.where(
    mask,
    1.0 - np.clip(dist_from_infection / max_dist, 0, 1),
    np.nan
)

# ── wind direction bias ───────────────────────────────────────────────────────
# Ethiopia's dominant wind during coffee growing season is SE
# Shift the proximity score slightly downwind (SE = positive x, positive y)
WIND_SHIFT_X = 8   # pixels eastward
WIND_SHIFT_Y = 8   # pixels southward

from scipy.ndimage import shift
wind_biased = shift(proximity_score, shift=[WIND_SHIFT_Y, WIND_SHIFT_X], cval=0)
wind_biased = np.where(mask, wind_biased, np.nan)

# ── smooth to get natural spread gradients ────────────────────────────────────
spread_risk = gaussian_filter(
    np.where(mask, wind_biased, 0), sigma=5
)
spread_risk = np.where(mask, spread_risk, np.nan)

# normalise to 0-100
spread_risk = np.where(
    mask,
    100 * (spread_risk - np.nanmin(spread_risk)) /
          (np.nanmax(spread_risk) - np.nanmin(spread_risk) + 1e-9),
    np.nan
)

# ── zero out already-infected zones ───────────────────────────────────────────
# spread risk only applies to currently healthy zones
spread_risk = np.where(seed_mask, np.nan, spread_risk)

print(f"\nSpread risk range: {np.nanmin(spread_risk):.1f}% → {np.nanmax(spread_risk):.1f}%")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: current infection probability
im1 = axes[0].imshow(prob, cmap=cmap_risk, vmin=0, vmax=100)
axes[0].set_title(f"Current infection probability — {dates[best_t].date()}")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="%")

# right: spread risk for healthy zones
combined = np.where(seed_mask, 101, spread_risk)   # 101 = confirmed, shown differently
cmap_spread = mcolors.LinearSegmentedColormap.from_list(
    "spread", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A"]
)
im2 = axes[1].imshow(spread_risk, cmap=cmap_spread, vmin=0, vmax=100)
axes[1].imshow(
    np.where(seed_mask, 1, np.nan),
    cmap=mcolors.ListedColormap(["#A32D2D"]),
    vmin=0, vmax=1
)
axes[1].set_title("Spread risk for currently healthy zones\n(red = confirmed infection source)")
axes[1].axis("off")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Spread risk %")

plt.tight_layout()
plt.savefig("stage2_spread_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: stage2_spread_prediction.png")

# ── summary stats for the farmer ──────────────────────────────────────────────
high_spread = (spread_risk >= 70) & mask & ~seed_mask
moderate_spread = (spread_risk >= 40) & (spread_risk < 70) & mask & ~seed_mask

print("\n── Farm Risk Summary ──────────────────────────────")
print(f"Confirmed/high infection area:  {seed_mask.sum() * 100 / 10000:.2f} ha")
print(f"High spread risk area (>70%):   {high_spread.sum() * 100 / 10000:.2f} ha")
print(f"Moderate spread risk (40-70%):  {moderate_spread.sum() * 100 / 10000:.2f} ha")
print("───────────────────────────────────────────────────")
print("Priority action: inspect and treat HIGH spread risk zones first")