# The Saudi Water Paradox: Flood Susceptibility Prediction (Jeddah)

**Final notebook: `jeddah_flood_final.ipynb`.** It runs the full workflow: loading the raw data, EDA, cleaning, machine learning and the final model.

| Folder / file | Content |
|---|---|
| `jeddah_flood_final.ipynb` | final notebook (Part 1: EDA, Part 2: machine learning) |
| `data/raw/` | raw datasets (`jeddah_flood_ml.csv`, `jeddah_flood_ml_elev_matched.csv`) |
| `data/processed/` | cleaned data and feature sets (created by the notebook) |
| `notebooks/02_eda.ipynb`, `notebooks/03_machine_learning.ipynb` | the two parts as separate notebooks |
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

## Web app
```bash
pip install -r requirements.txt
streamlit run app.py
```
To put it online for free: push this folder to GitHub, go to https://share.streamlit.io, choose the repository and set the main file to `app.py`.
