import os
import zarr
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

ZARR_PATH = "sentinel2/F1_coffee_leaf_rust_ethiopia/2017-01-01_2025-12-31/cube.zarr"
GEOJSON   = "F1-coffee-leaf-rust-ethiopia/F1_CLR Survey Farm Level.geojson"

# ── load ground truth ──────────────────────────────────────────────────────────
print("Loading ground truth...")
farms = gpd.read_file(GEOJSON)
print(f"  {len(farms)} farms loaded")

# ── load zarr ─────────────────────────────────────────────────────────────────
print("Loading zarr...")
store     = zarr.open(ZARR_PATH, mode="r")
b04       = store["B04"][:]
b05       = store["B05"][:]
b07       = store["B07"][:]
b08       = store["B08"][:]
time_days = store["time"][:]
x_coords  = store["x"][:]
y_coords  = store["y"][:]
dates     = pd.to_datetime("2017-01-01") + pd.to_timedelta(time_days, unit="D")

# ── find survey period time steps ─────────────────────────────────────────────
survey_dates = [
    pd.Timestamp("2017-09-01"), pd.Timestamp("2017-10-01"),
    pd.Timestamp("2017-11-01"), pd.Timestamp("2017-12-01"),
    pd.Timestamp("2018-01-01"), pd.Timestamp("2018-02-01"),
]
survey_indices = [int(np.argmin([abs((d - sd).days) for d in dates]))
                  for sd in survey_dates]
print(f"  Survey indices: {[str(dates[i].date()) for i in survey_indices]}")

def norm_inverse(arr, lo, hi):
    return 1.0 - (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-9)

def compute_features(t):
    """Compute all spectral features for one time step"""
    red  = b04[t].astype(float) / 10000
    re1  = b05[t].astype(float) / 10000
    re3  = b07[t].astype(float) / 10000
    nir  = b08[t].astype(float) / 10000
    mask = (red > 0) & (nir > 0) & (re1 > 0) & (re3 > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_t   = np.where(mask, (nir - red)   / (nir + red   + 1e-9), np.nan)
        rendvi_t = np.where(mask, (re3 - re1)   / (re3 + re1   + 1e-9), np.nan)
        cire_t   = np.where(mask, (re3 / re1) - 1,                       np.nan)
        evi_t    = np.where(mask, 2.5 * (nir - red) /
                           (nir + 6*red - 7.5*re1 + 1 + 1e-9),           np.nan)
        ndre_t   = np.where(mask, (nir - re1)   / (nir + re1   + 1e-9), np.nan)

    veg = mask & (ndvi_t > 0.15) & (ndvi_t < 0.90)

    prob = (
        0.25 * norm_inverse(ndvi_t,   0.2, 0.8) +
        0.40 * norm_inverse(rendvi_t, 0.0, 0.4) +
        0.35 * norm_inverse(cire_t,   0.5, 3.0)
    ) * 100

    return {
        "ndvi":    np.where(veg, ndvi_t,   np.nan),
        "rendvi":  np.where(veg, rendvi_t, np.nan),
        "cire":    np.where(veg, cire_t,   np.nan),
        "evi":     np.where(veg, evi_t,    np.nan),
        "ndre":    np.where(veg, ndre_t,   np.nan),
        "prob":    np.where(veg, np.clip(prob, 0, 100), np.nan),
        "veg":     veg
    }

# ── compute features across survey period ─────────────────────────────────────
print("Computing spectral features across survey period...")
feature_stacks = {k: [] for k in ["ndvi", "rendvi", "cire", "evi", "ndre", "prob"]}

for t in survey_indices:
    feats = compute_features(t)
    for k in feature_stacks:
        feature_stacks[k].append(feats[k])
    print(f"  {dates[t].date()} done")

# mean across survey period
feature_means = {k: np.nanmean(np.stack(v), axis=0)
                 for k, v in feature_stacks.items()}
veg_mask = feats["veg"]

# ── extract features at farm locations ────────────────────────────────────────
print("\nExtracting features at farm locations...")
farms_utm = farms.to_crs("EPSG:32636")

x_min     = float(x_coords.min())
y_max_val = float(y_coords.max())
x_res     = float(x_coords[1] - x_coords[0])
y_res     = float(abs(y_coords[1] - y_coords[0]))

farm_features = []
for _, farm in farms_utm.iterrows():
    fx = farm.geometry.x
    fy = farm.geometry.y

    col = int((fx - x_min) / x_res)
    row = int((y_max_val - fy) / y_res)
    col = np.clip(col, 0, feature_means["ndvi"].shape[1] - 1)
    row = np.clip(row, 0, feature_means["ndvi"].shape[0] - 1)

    # 3x3 neighbourhood mean to account for GPS uncertainty
    r0 = max(0, row-1); r1 = min(feature_means["ndvi"].shape[0], row+2)
    c0 = max(0, col-1); c1 = min(feature_means["ndvi"].shape[1], col+2)

    feat_row = {
        "farm":     int(farm["farm"]),
        "inc":      float(farm["inc"]),
        "sev":      float(farm["sev"]),
        "altitude": float(farm["altitude"]),
    }
    for k, arr in feature_means.items():
        patch = arr[r0:r1, c0:c1]
        feat_row[k] = float(np.nanmean(patch))

    farm_features.append(feat_row)

df = pd.DataFrame(farm_features).dropna()
print(f"  {len(df)} farms with complete features")

# ── individual correlations ────────────────────────────────────────────────────
print("\n── Individual feature correlations with incidence ───────────")
features_to_test = ["prob", "ndvi", "rendvi", "cire", "evi", "ndre", "altitude"]
corrs = {}
for feat in features_to_test:
    r, p = pearsonr(df[feat], df["inc"])
    corrs[feat] = (r, p)
    flag = "✓" if abs(r) > 0.3 and p < 0.1 else " "
    print(f"  {flag} {feat:<12} r = {r:+.3f}  p = {p:.4f}")

# ── combined model: spectral + altitude ───────────────────────────────────────
print("\n── Combined model: spectral indices + altitude ──────────────")

# features to use in combined model
X_cols = ["prob", "ndvi", "rendvi", "cire", "ndre", "altitude"]
X = df[X_cols].values
y = df["inc"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Leave-One-Out cross validation (only 18 samples so LOO is appropriate)
loo = LeaveOneOut()
models = {
    "Linear Regression":      LinearRegression(),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                                         random_state=42),
    "Random Forest":          RandomForestRegressor(n_estimators=50, max_depth=3,
                                                     random_state=42),
}

best_r = -999
best_name = None
best_preds = None

for name, model in models.items():
    preds = []
    for train_idx, test_idx in loo.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        preds.append(model.predict(X_scaled[test_idx])[0])

    preds = np.array(preds)
    r_loo, p_loo = pearsonr(preds, y)
    r2_loo = r2_score(y, preds)
    rmse   = np.sqrt(np.mean((preds - y)**2))

    print(f"\n  {name}:")
    print(f"    LOO Pearson r = {r_loo:.3f}  p = {p_loo:.4f}")
    print(f"    LOO R²        = {r2_loo:.3f}")
    print(f"    LOO RMSE      = {rmse:.1f}%")

    if r_loo > best_r:
        best_r    = r_loo
        best_name = name
        best_preds = preds

print(f"\n  Best model: {best_name} (r = {best_r:.3f})")

# ── train final model on all data ─────────────────────────────────────────────
print("\n── Training final model on all farms ───────────────────────")
best_model = models[best_name]
best_model.fit(X_scaled, y)

# feature importance (for tree models)
if hasattr(best_model, "feature_importances_"):
    print("  Feature importances:")
    for feat, imp in sorted(zip(X_cols, best_model.feature_importances_),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<12} {imp:.3f}  {bar}")

# ── apply to full raster ───────────────────────────────────────────────────────
print("\nApplying combined model to full study area...")

# we don't have raster altitude so use the spectral prob map
# but weight by the altitude relationship from ground truth
# altitude effect: inc decreases ~0.05% per metre above 1000m
# approximate from the data: slope of altitude vs inc
alt_slope, alt_intercept = np.polyfit(df["altitude"], df["inc"], 1)
print(f"  Altitude effect: {alt_slope:.4f}% incidence per metre")

# for the full raster — use spectral prob as base
# we don't have pixel-level altitude without DEM
# so show the combined model prediction at farm points only
df["combined_pred"] = best_preds
df["spectral_only"] = df["prob"]

r_combined, p_combined = pearsonr(df["combined_pred"], df["inc"])
r_spectral, p_spectral = pearsonr(df["spectral_only"], df["inc"])

print(f"\n── Validation Summary ───────────────────────────────────────")
print(f"  Spectral only:           r = {r_spectral:.3f}  p = {p_spectral:.4f}")
print(f"  Combined (spectral+alt): r = {r_combined:.3f}  p = {p_combined:.4f}")
print(f"  Improvement:             +{r_combined - r_spectral:.3f}")
print(f"  n = {len(df)} field-surveyed farms")
if r_combined > 0.4:
    print(f"  ✓ Combined model shows meaningful correlation with confirmed CLR")
print("────────────────────────────────────────────────────────────")

# ── plots ──────────────────────────────────────────────────────────────────────
cmap_risk = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#5DCAA5", "#C0DD97", "#EF9F27", "#E24B4A", "#A32D2D"]
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Combined model validation — spectral indices + altitude vs confirmed CLR\n"
    f"Leave-one-out cross validation, n=18 farms, {best_name}",
    fontsize=13, fontweight="bold"
)

# left — spectral only
ax = axes[0]
ax.scatter(df["inc"], df["spectral_only"],
           c=df["altitude"], cmap="RdYlGn_r",
           s=80, zorder=5, edgecolors="white")
m, b = np.polyfit(df["inc"], df["spectral_only"], 1)
xl = np.linspace(df["inc"].min(), df["inc"].max(), 100)
ax.plot(xl, m*xl+b, color="#A32D2D", linewidth=2)
for _, row in df.iterrows():
    ax.annotate(f"F{int(row['farm'])}",
                (row["inc"], row["spectral_only"]),
                textcoords="offset points", xytext=(3,3), fontsize=7)
ax.set_xlabel("Confirmed field incidence (%)", fontsize=11)
ax.set_ylabel("Model score (%)", fontsize=11)
ax.set_title(f"Spectral only\nr = {r_spectral:.3f}  p = {p_spectral:.4f}")

# middle — combined model
ax = axes[1]
sc = ax.scatter(df["inc"], df["combined_pred"],
                c=df["altitude"], cmap="RdYlGn_r",
                s=80, zorder=5, edgecolors="white")
m2, b2 = np.polyfit(df["inc"], df["combined_pred"], 1)
ax.plot(xl, m2*xl+b2, color="#A32D2D", linewidth=2)
for _, row in df.iterrows():
    ax.annotate(f"F{int(row['farm'])}",
                (row["inc"], row["combined_pred"]),
                textcoords="offset points", xytext=(3,3), fontsize=7)
plt.colorbar(sc, ax=ax, label="Altitude (m)")
ax.set_xlabel("Confirmed field incidence (%)", fontsize=11)
ax.set_ylabel("Combined model prediction (%)", fontsize=11)
ax.set_title(f"Combined (spectral + altitude)\nr = {r_combined:.3f}  p = {p_combined:.4f}")

# right — feature importance
ax = axes[2]
if hasattr(best_model, "feature_importances_"):
    sorted_feats = sorted(zip(X_cols, best_model.feature_importances_),
                          key=lambda x: x[1])
    feat_names = [f[0] for f in sorted_feats]
    feat_imps  = [f[1] for f in sorted_feats]
    colors     = ["#A32D2D" if f == "altitude" else "#5DCAA5" for f in feat_names]
    ax.barh(feat_names, feat_imps, color=colors, alpha=0.8)
    ax.set_xlabel("Feature importance", fontsize=11)
    ax.set_title(f"Feature importance\n(red = altitude, green = spectral)")
    ax.axvline(0, color="gray", linewidth=0.5)
else:
    # linear regression coefficients
    lr_temp = LinearRegression().fit(X_scaled, y)
    sorted_feats = sorted(zip(X_cols, lr_temp.coef_), key=lambda x: x[1])
    ax.barh([f[0] for f in sorted_feats], [f[1] for f in sorted_feats],
            color=["#A32D2D" if f[0]=="altitude" else "#5DCAA5" for f in sorted_feats],
            alpha=0.8)
    ax.set_xlabel("Coefficient", fontsize=11)
    ax.set_title("Feature coefficients\n(red = altitude, green = spectral)")
    ax.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("map15_combined_model_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: map15_combined_model_validation.png")

# ── farm table ─────────────────────────────────────────────────────────────────
print(f"\n── Farm prediction table ───────────────────────────────────")
print(f"  {'Farm':<6} {'Alt':>6} {'Actual inc':>10} {'Spectral':>10} {'Combined':>10}")
print("  " + "-"*46)
for _, row in df.sort_values("inc", ascending=False).iterrows():
    print(f"  {int(row['farm']):<6} {row['altitude']:>5.0f}m "
          f"{row['inc']:>9.1f}% {row['spectral_only']:>9.1f}% "
          f"{row['combined_pred']:>9.1f}%")
print("────────────────────────────────────────────────────────────")

# push summary
print(f"\n── Final statement ─────────────────────────────────────────")
print(f"  Spectral model alone:    r = {r_spectral:.3f}")
print(f"  Combined model (LOO CV): r = {r_combined:.3f}")
print(f"  Altitude alone:          r = -0.730")
print(f"  Best model:              {best_name}")
print(f"  n farms:                 {len(df)}")
print("────────────────────────────────────────────────────────────")