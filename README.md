## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app)

**Try the app:** https://jeddah-flood.streamlit.app/

The app lets you:
- play a guessing game against the model,
- build a piece of land and see its flood susceptibility,
- explore a 3D storm map of Jeddah.

> The model gives a *relative* flood-susceptibility score for open-land cells, based on where satellite radar detected water after two storms (2017 and 2022). It is not a real-time warning system and it does not predict water depth.

If the app is asleep, click **"Yes, get this app back up!"** and wait about a minute.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
# The Saudi Water Paradox: Flood Susceptibility Prediction (Jeddah)

**Final notebook: `jeddah_flood_final.ipynb`.** It runs the full workflow: loading the raw data, EDA, cleaning, machine learning and the final model.

| Folder / file | Content |
|---|---|
| `jeddah_flood_final.ipynb` | final notebook (Part 1: EDA, Part 2: machine learning) |
| `data/processed/` | cleaned data and feature sets (created by the notebook) |
| `src/preprocessing.py` | preprocessing helpers (created by the EDA part) |
| `app.py` | Streamlit web app: a guessing game against the model, a land builder and a 3D storm map |
| `app_data/`, `scripts/make_app_data.py` | small files for the app (map points, game squares, examples) and the script that makes them |
| `src/predict.py` | `load_model()` and `predict_susceptibility()` for the Streamlit app |
| `models/flood_model.joblib` | final Random Forest model |
| `outputs/` | metrics tables and figures |
| `docs/` | data sources, handover, QA report, EDA findings, ML evaluation |

## Run
Open the notebook from the main project folder (where this README is):

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
jupyter notebook jeddah_flood_final.ipynb   # about 8-10 minutes
```

The model was saved with scikit-learn 1.9.1. With another version, re-run the notebook to rebuild it.
