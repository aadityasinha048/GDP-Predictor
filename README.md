# GDP Prediction Model

## Overview
This project predicts GDP using macroeconomic indicators from World Bank data.

## Features
- Hybrid missing value imputation
- Advanced feature selection (Correlation + MI + RF + RFE)
- Multiple ML models (Linear, Tree, Boosting)
- Country-aware modeling

## Project Structure
- `src/` → core pipeline
- `data/` → datasets
- `models/` → saved models
- `notebooks/` → experiments

## Run
```bash
pip install -r requirements.txt
python src/train.py