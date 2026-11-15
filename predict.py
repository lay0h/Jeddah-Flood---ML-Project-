"""Use the trained flood-susceptibility model (for example inside Streamlit).

The model gives a RELATIVE susceptibility score (0-1) for a 30 m cell in Jeddah:
how much the cell's terrain looks like places where Sentinel-1 saw flood water
remain 4-5 days after the 2017 and 2022 storms. It does NOT predict water depth,
and the score is NOT a calibrated probability (training data was sampled 1:1).

Streamlit example:

    import streamlit as st
    from src.predict import load_model, predict_susceptibility

    @st.cache_resource
    def get_model():
        return load_model("models/flood_model.joblib")

    bundle = get_model()
    result = predict_susceptibility(input_df, bundle)
    st.dataframe(result[["susceptibility_score", "susceptibility_class", "warning"]])
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Raw columns a user (or the app) must provide for each cell
RAW_INPUTS = ["elevation_m", "slope_deg", "flowacc_km2", "twi", "hand_m",
              "dist_to_coast_m", "dist_to_ref_water_m", "landcover_name"]

# Same grouping of land cover classes as in the EDA notebook (02_eda.ipynb)
LANDCOVER_GROUP = {"bare_sparse": "bare_sparse", "built_up": "built_up",
                   "tree_cover": "vegetation", "shrubland": "vegetation",
                   "grassland": "vegetation", "cropland": "vegetation",
                   "permanent_water": "water_wetland", "herbaceous_wetland": "water_wetland",
                   "mangroves": "water_wetland"}


def load_model(path="models/flood_model.joblib"):
    """Load the saved bundle: pipeline, feature list, threshold and metadata."""
    return joblib.load(Path(path))


def add_model_features(df):
    """Create the model inputs from raw columns (same transformations as the EDA)."""
    out = df.copy()
    if "flowacc_log10" not in out:
        out["flowacc_log10"] = np.log10(out["flowacc_km2"])
    if "hand_log1p" not in out:
        out["hand_log1p"] = np.log1p(out["hand_m"])
    if "dist_coast_log1p" not in out:
        out["dist_coast_log1p"] = np.log1p(out["dist_to_coast_m"])
    if "dist_refwater_log1p" not in out:
        out["dist_refwater_log1p"] = np.log1p(out["dist_to_ref_water_m"])
    if "landcover_group" not in out:
        out["landcover_group"] = out["landcover_name"].map(LANDCOVER_GROUP)
    return out


def predict_susceptibility(df, bundle):
    """Return the input rows plus a score, a class and a warning column.

    susceptibility_class: 'higher' if score >= the model threshold, else 'lower'.
    warning: flags rows where the model should not be trusted
             (built-up land: the radar labels cannot see water in the city;
              values outside the training range; missing inputs).
    """
    features = bundle["features"]
    data = add_model_features(df)
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}. Required raw inputs: {RAW_INPUTS}")

    score = bundle["pipeline"].predict_proba(data[features])[:, 1]
    result = df.copy()
    result["susceptibility_score"] = np.round(score, 3)
    result["susceptibility_class"] = np.where(score >= bundle["threshold"], "higher", "lower")

    warnings = pd.Series("", index=df.index)
    warnings[data["landcover_group"] == "built_up"] += "built-up area: labels blind here; "
    for col, (low, high) in bundle["training_range"].items():
        outside = (data[col] < low) | (data[col] > high)
        warnings[outside] += f"{col} outside training range; "
    warnings[data[features].isna().any(axis=1)] += "missing input imputed; "
    result["warning"] = warnings.str.strip().str.rstrip(";")
    return result
