# EDA Findings: Jeddah Flood Dataset

The dataset covers two flood events (21 Nov 2017 and 24 Nov 2022). All numbers below were computed in `notebooks/02_eda.ipynb`, and the findings describe these two events only.

## 1. Data
- Main file: 40,054 rows (15,092 in 2017 and 24,962 in 2022). Elevation-matched file: 35,844 rows.
- Share of flooded cells: 50% (main) and 56% (matched). This comes from the sampling design and is not the real flood prevalence, so precision and PR-AUC will not reflect real-world performance.
- No duplicate rows and no physically impossible values.

## 2. Missing values
- `slope_deg` and `twi`: 55 missing in the main file and 473 in the matched file (mostly dry cells). `hand_m`: 738 missing in the main file. All other model inputs are complete.
- Handling: median imputation inside the modeling pipeline (fitted on training data only). No "is missing" features, because they would leak the label.

## 3. Elevation
- Median elevation in the main file: dry 20.0 m, flooded 0.56 m. In the matched file: dry 0.67 m, flooded 0.56 m.
- Single-feature ROC-AUC of elevation (mean of the two events): 0.04 in the main file and 0.41 in the matched file. Every model should be compared to an elevation-only baseline.

## 4. Land cover
- Flooded cells are almost all bare or sparse land. Built-up areas are rare among flooded cells because radar cannot see water between buildings. This is a limit of the labels and does not mean that cities are safe.

## 5. Rainfall and soil moisture
- IMERG rainfall has only about 24 distinct values per event (about 11 km resolution). In the main file, 24 h rainfall and the maximum 30-min rate are lower at flooded cells, and in the matched file the direction of the single-feature AUC changes between events. They probably act as location indicators (interpretation, not proof).
- `rain_upstream_event_mm` is almost identical to `rain_event_24h_mm` and is not used.
- Models should be tested with and without the rainfall features.

## 6. Spatial structure
- The top 10 blocks of 1 km hold 47% (2017) and 37% (2022) of flooded cells.
- 5,456 cells appear in both events and 5,166 of them were flooded in both. A random split would leak information, so we split by location (`block_5km`) or by event.
- Spatial folds are unbalanced (see `outputs/spatial_folds_check.csv`), so results should be reported per fold.

## 7. Outliers
- Extreme values (high ground, long flow paths, far from the coast) are real and were kept. Log versions of the skewed columns are included.

## 8. Notes for modeling
1. Use spatial cross-validation on `block_5km` and also test 2017 to 2022 and 2022 to 2017.
2. Evaluate on the elevation-matched file and report an elevation-only baseline next to every model.
3. Compare the feature sets `terrain`, `terrain_soil` and `terrain_soil_rain`.
4. Report ROC-AUC, recall, F1 and the confusion matrix, never accuracy alone.

## 9. Limitations
- Radar sees little water inside the city, so labels come mostly from open land and the coast.
- Labels come from images taken 4-5 days after the events, so they show remaining water and not the maximum flood extent.
- Land cover is from 2021 for both events. Correlation does not mean causation.
