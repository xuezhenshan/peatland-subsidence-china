# National-scale Modeling of Peatland Subsidence in China

This repository contains R and Python code for the national modeling and prediction of peatland subsidence, distribution, and soil properties in China.  
It accompanies the manuscript:  
**"Climate–human interactions drive widespread peatland subsidence and carbon vulnerability in China"**

## Structure

### R Scripts

- `R/peatland_profile_prediction.R`  
  Predicts peat depth, total organic carbon (TOC), and bulk density using Random Forest.

- `R/peatland_distribution_model.R`  
  Builds an ensemble model (RF, GBM, XGBoost) for national peatland distribution, constrained by depth and TOC.

### Python Script

- `Python/peatland_subsidence_SHAP.py`  
  Trains an XGBoost model on SBAS-InSAR data to predict national peatland subsidence and calculates SHAP values.

### Data

- `example_data/sample_predictors.tif` — Environmental rasters (subset)  
- `example_data/peat_profiles.csv` — Peat core samples for model calibration

## Requirements

### For R scripts

R packages:  
`randomForest`, `xgboost`, `gbm`, `raster`, `rgdal`, `dplyr`, `sp`, `sf`, `caret`

### For Python script

Python ≥ 3.8  
```bash
pip install xgboost shap rasterio geopandas pandas scikit-learn matplotlib
```

## How to run

1. Place all environmental predictor layers in a directory.
2. For R scripts, modify file paths in the scripts and run in RStudio.
3. For Python:
   ```bash
   cd Python
   python peatland_subsidence_SHAP.py
   ```

## Citation

> Xue et al. (2025). *Climate–human interactions drive widespread peatland subsidence and carbon vulnerability in China*. (Submitted)

## License

MIT License

## Contact

Zhenshan Xue — xueshenshan@iga.ac.cn
