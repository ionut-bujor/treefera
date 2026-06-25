# Treefera — Coffee Leaf Rust Early Warning System

Satellite-driven detection of *Hemileia vastatrix* (Coffee Leaf Rust) in pre-symptomatic
stages across 10,740 ha of forest coffee in Southwest Ethiopia, built at the
**Treefera LCAW 2026 Hackathon** (London Climate Week, BlueFin Building, 25 June 2026).

 **People's Choice Award** — voted by Treefera employees

---

## The Problem

Coffee Leaf Rust (CLR) costs the global coffee industry $1–2 billion annually. In Ethiopia —
one of the world's most important origins — farmers detect it by walking their fields
looking for yellow-orange pustules on leaf undersides. By the time symptoms are visible,
the fungus has been present for ~20 days and is already spreading.

**Can satellites detect CLR before farmers can see it?**

---

## What This Does

The pipeline detects early chlorophyll breakdown in the red-edge spectral bands of
Sentinel-2 imagery — a signal that appears 2–3 weeks before visible symptoms. It then
models where infection is likely to spread next.

The core insight: RENDVI and CIre drop when CLR attacks chlorophyll, while NDVI still
looks healthy. That gap is the detection window.

### Stage 1 — Infection Probability

Weighted spectral score across vegetation pixels:

| Index | Weight | What it catches |
|-------|--------|----------------|
| RENDVI | 40% | Primary pre-symptomatic signal |
| CIre | 35% | Early chlorophyll loss |
| NDVI | 25% | General stress confirmation |

**Results on 1 May 2024:**
- Pre-symptomatic CLR: **3.86 ha**
- Symptomatic CLR: **3.05 ha**
- Confirmed / high infection: **19.69 ha**

### Stage 2 — Spread Prediction

Distance transform from confirmed infection zones, SE wind direction bias,
Gaussian smoothing (σ=25). Tells farmers *where* to act next.

- High spread risk (>70%): **5,036 ha**
- Moderate spread risk (40–70%): **3,555 ha**

---

## Data Stack

| Sensor | Resolution | Role |
|--------|-----------|------|
| Sentinel-2 | 10m | Primary detection — spectral indices |
| Sentinel-1 | 10m | Soil moisture — drought confound ruled out |
| PALSAR L-band (ALOS-2) | 25m | Canopy penetration — structural validation |
| SPOT 6 | 1.5m | Visual validation at individual tree crown level |
| Forest Data Partnership coffee model | 10m | Coffee land use masking |
| Belachew et al. (2020) field survey | GPS points | Ground truth — 18 farms, 32.5–86.7% incidence |

8 years of Sentinel-2 data (January 2017 – December 2025), 108 monthly composites.

---

## Validation — Five Independent Lines of Evidence

### 1. Spatial Correlation
Model probability vs subsequent NDVI decline: **r=0.628**, p=0.0000 across 1,059,075
vegetation pixels.

### 2. Temporal Outbreak Detection
Running the model blind across September 2019–June 2021, it independently peaked in
**March 2021** — matching a documented CLR outbreak in Jimma zone confirmed by KU Leuven
and Jimma University field surveys (Daba et al., Heliyon 2022). Stress +22.1% above baseline.

### 3. Drought Ruled Out
Sentinel-1 VV backscatter confirms 2020 was the **wettest year** in the 2017–2025 dataset
(mean VV: 0.846). OCHA flood reports confirm flooding in Jimma/Illubabor during the 2020
Kiremt season. Wet infection year → stress peak = CLR, not drought stress.

### 4. Seasonal Fingerprint
January–March mean stress score: **35.9%** vs all other months: **22.8%**.
T-test: t=4.233, **p=0.0000** across 8 years. This repeating biological cycle matches
the documented CLR seasonal signature in Ethiopia.

### 5. Ground Truth Validation
Combined Gradient Boosting model validated against 18 field-surveyed farms:

| Model | LOO Pearson r | p-value | RMSE |
|-------|-------------|---------|------|
| Spectral only | 0.176 | 0.4839 | — |
| Linear Regression | 0.551 | 0.0179 | 15.3% |
| Random Forest | 0.673 | 0.0022 | 11.7% |
| **Gradient Boosting** | **0.776** | **0.0002** | **10.4%** |

Altitude confirmed as dominant driver: r=−0.730 (p=0.0006), consistent with Belachew et al. (2020).

---

## Repository Structure

```
treefera/
├── clr_pipeline.py              # Main pipeline: spectral indices, probability, spread
├── infection_probability.py     # Extended infection scoring
├── spread_prediction.py         # Wind-biased spread model
├── sentinel2/                   # Zarr data cube (not included — see Data section)
└── README.md
```

Additional scripts used in analysis (outputs not committed):

| Script | Purpose |
|--------|---------|
| `validation_timeseries.py` | 2019–2021 outbreak detection, seasonal fingerprint |
| `sentinel1_integration.py` | Soil moisture and drought ruling |
| `smoking_gun_validation.py` | Full 8-year time series, seasonal t-test |
| `ground_truth_validation.py` | 18-farm survey vs model |
| `combined_model.py` | Gradient Boosting LOO CV |
| `coffee_mask_integration.py` | Forest Data Partnership coffee masking |
| `palsar_integration.py` | PALSAR L-band HV backscatter |
| `spot_analysis.py` | SPOT 6 1.5m canopy change 2020→2024 |
| `build_presentation.py` | HTML dashboard |
| `run_all.sh` | Master runner |

---

## Running the Pipeline

```bash
# Clone and set up environment
git clone https://github.com/ionut-bujor/treefera.git
cd treefera
python -m venv venv && source venv/bin/activate
pip install zarr numpy matplotlib pandas scipy scikit-learn rasterio geopandas

# Run main pipeline (requires Sentinel-2 Zarr cube at sentinel2/ path)
python clr_pipeline.py
```

Output: `clr_analysis.png` — four-panel map showing NDVI, RENDVI, infection probability,
and spread risk.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Study area | 10,740 ha, SW Ethiopia |
| Time series | 8 years (2017–2025), 108 time steps |
| Pre-symptomatic CLR detected | 3.86 ha |
| Confirmed infection | 19.69 ha |
| High spread risk | 5,036 ha |
| Spatial validation | r=0.628, 1M+ pixels |
| Ground truth validation | r=0.776, p=0.0002 |
| Temporal validation | March 2021 blind peak = documented outbreak |
| Altitude vs incidence | r=−0.730, p=0.0006 |
| PALSAR HV at CLR zones | −1.77 dB vs healthy |
| Sensors used | 5 (S2, S1, PALSAR, SPOT 6, coffee probability model) |

---

## Honest Limitations

- **Spectral model alone is weak at farm level** (r=0.176) — altitude does 92% of the work in the combined model
- **Cannot see through canopy** — Ethiopia's forest coffee grows under closed Afromontane canopy; optical sensors see the canopy, not the plants underneath
- **Cannot distinguish CLR from CBD** — Coffee Berry Disease produces similar spectral signatures at 10m resolution
- **6-year temporal gap** between ground truth surveys (2017–2018) and main analysis imagery (2024)
- **Spread model simplified** — Gaussian kernel may overestimate immediate at-risk zone; ERA5 wind fields would improve precision

---

## What Would Make This Production Ready

1. DEM integration — SRTM elevation as a full raster layer for altitude-weighted probability maps
2. ERA5 wind fields — replace fixed SE wind assumption with real reanalysis data
3. GPS calibration campaign — field visits to flagged zones to validate thresholds
4. Weekly compositing — reduce detection lag from 4 weeks to 7 days
5. ALOS-2 PALSAR time series — multi-year L-band for structural canopy change tracking
6. Larger ground truth dataset — 100+ farms across altitude bands and seasons

---

## Study Area

**Location:** Jimma and Illubabor zones, Southwest Ethiopia
**Coffee system:** Forest coffee — plants grow under dense Afromontane canopy
**Ground truth:** Belachew K. et al. (2020), OSF doi:10.31219/osf.io/gdr5v

---

## Built at

**Treefera LCAW 2026 Hackathon** — "The Realm of the Impossible"
London Climate Week, BlueFin Building, London, 25 June 2026

Challenge: Cacao Black Pod Detection / Coffee Leaf Rust Detection
