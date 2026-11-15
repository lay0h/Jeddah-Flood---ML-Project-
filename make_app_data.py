"""Prepare the small files the Streamlit app needs (run once from the project root).

1. app_data/map_points.csv        - every sampled cell with a model score from spatial cross-validation
                                    (each cell is scored by a model that never saw its 5 km block)
2. app_data/game_cells.csv        - cells for the guessing game (easy and hard level)
3. app_data/examples.json          - a few real cells used as examples in the app
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from preprocessing import make_preprocessor

OUT = ROOT / 'app_data'
OUT.mkdir(exist_ok=True)
features = json.load(open(ROOT / 'data/processed/feature_sets.json'))['feature_sets']['terrain']
datasets = {'main': pd.read_csv(ROOT / 'data/processed/jeddah_flood_clean.csv'),
            'matched': pd.read_csv(ROOT / 'data/processed/jeddah_flood_elev_matched_clean.csv')}


def make_rf():   # same settings as the notebook
    return Pipeline([('prep', make_preprocessor(features)),
                     ('clf', RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features='sqrt',
                                                    n_jobs=-1, random_state=42))])


folds = {name: list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(df, df.label, groups=df.block_5km))
         for name, df in datasets.items()}

# 1. map points (main file, spatial CV)
main = datasets['main']
score = np.zeros(len(main))
for train_idx, test_idx in folds['main']:
    model = make_rf().fit(main.iloc[train_idx][features], main.label.iloc[train_idx])
    score[test_idx] = model.predict_proba(main.iloc[test_idx][features])[:, 1]
points = main[['lon', 'lat', 'year', 'label', 'elevation_m', 'landcover_group']].copy()
points['score'] = score.round(3)
points[['lon', 'lat']] = points[['lon', 'lat']].round(5)
points['elevation_m'] = points.elevation_m.round(1)
points.to_csv(OUT / 'map_points.csv', index=False)
print('map points:', len(points))

# 1b. cells for the guessing game: easy = main file, hard = elevation-matched file (scores from spatial CV)
matched = datasets['matched']
score_m = np.zeros(len(matched))
for train_idx, test_idx in folds['matched']:
    model = make_rf().fit(matched.iloc[train_idx][features], matched.label.iloc[train_idx])
    score_m[test_idx] = model.predict_proba(matched.iloc[test_idx][features])[:, 1]
game = []
for level, df, sc in [('easy', main, score), ('hard', matched, score_m)]:
    g = df[['lon', 'lat', 'year', 'label', 'elevation_m', 'slope_deg', 'dist_to_coast_m', 'landcover_group']].assign(score=sc.round(3), level=level)
    g = g.dropna().groupby('label').sample(200, random_state=42)       # 200 flooded + 200 dry per level
    print(level, 'model accuracy on game cells:', ((g.score >= 0.5).astype(int) == g.label).mean().round(2))
    game.append(g)
game = pd.concat(game).round({'lon': 5, 'lat': 5, 'elevation_m': 1, 'slope_deg': 1, 'dist_to_coast_m': 0})
game.to_csv(OUT / 'game_cells.csv', index=False)

# 2. example cells (real rows from the 2022 storm)
raw_cols = ['elevation_m', 'slope_deg', 'flowacc_km2', 'twi', 'hand_m', 'dist_to_coast_m', 'dist_to_ref_water_m', 'landcover_name']
d = main[(main.year == 2022)].dropna(subset=raw_cols)
picks = {
    'Southern coastal flats': d[(d.lat.between(21.30, 21.34)) & (d.label == 1) & (d.landcover_name == 'bare_sparse')],
    'North of Obhur': d[(d.lat > 21.80) & (d.label == 1) & (d.landcover_name == 'bare_sparse')],
    'Inland high ground': d[(d.elevation_m.between(60, 120)) & (d.label == 0) & (d.landcover_name == 'bare_sparse')],
    'Inside the city': d[(d.landcover_name == 'built_up') & (d.label == 0) & (d.elevation_m.between(5, 20))],
}
examples = {}
for name, rows_ in picks.items():
    r = rows_.sort_values('elevation_m').iloc[len(rows_) // 2]          # a typical (median elevation) cell
    examples[name] = {**{c: (r[c] if c == 'landcover_name' else round(float(r[c]), 3)) for c in raw_cols},
                      'lat': round(float(r.lat), 4), 'lon': round(float(r.lon), 4), 'seen_flooded_2022': int(r.label)}
json.dump(examples, open(OUT / 'examples.json', 'w'), indent=2)
print(json.dumps(examples, indent=1))
