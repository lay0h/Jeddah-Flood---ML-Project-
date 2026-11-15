import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ['landcover_group']


def load_feature_sets(path='data/processed/feature_sets.json'):
    return json.load(open(path))['feature_sets']


def make_preprocessor(features, scale=False):
    # unfitted preprocessing; use scale=True for Logistic Regression
    numeric = [c for c in features if c not in CATEGORICAL]
    categorical = [c for c in features if c in CATEGORICAL]
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    parts = [('num', Pipeline(numeric_steps), numeric)]
    if categorical:
        parts.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))
    return ColumnTransformer(parts)


def spatial_folds(df, group_col='block_5km', n_splits=5, seed=42):
    # cross-validation where each spatial block stays entirely in one fold
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, df['label'], groups=df[group_col]))


def event_split(df, train_year, test_year):
    # train on one flood event and test on the other
    return np.where(df.year == train_year)[0], np.where(df.year == test_year)[0]
