# Land Price Predictor (Omakotitalotontit, Finland)

Predicts plot €/m² using StatFin quarterly data (2015–2025).

## Project Structure

- `data/` → raw and processed datasets  
- `notebooks/` → Jupyter notebooks for exploration (EDA)  
- `src/` → preprocessing, model training, prediction class  
- `api/` → FastAPI backend for serving predictions  
- `models/` → saved model files and scaler parameters  
- `logs/` → CSV logs of all predictions  

## Workflow

1. Exploratory Data Analysis (EDA)  
2. Preprocessing & Feature Engineering  
3. Model Training & Evaluation  
4. Saving Model & Scaling Parameters  
5. Serving Predictions via FastAPI
