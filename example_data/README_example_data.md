# Example Dataset for Peatland Subsidence Modeling in China

This dataset provides example environmental variables used in the study *"Climate–human interactions drive widespread peatland subsidence and carbon vulnerability in China"*. The variables have been resampled to ~5 km resolution for demonstration and testing purposes only.

## 📁 Directory Structure

```
example_data/
├── peatland_distribution/    # Variables used for peatland distribution modeling
├── peat_profile/             # Variables used for peat depth, TOC, and BD modeling
└── subsidence/               # Variables used for peatland subsidence prediction
    ├── current/              # Current baseline scenario (2017–2023)
    ├── SSP126/               # SSP1-2.6 climate scenario
    ├── SSP370/               # SSP3-7.0 climate scenario
    └── SSP585/               # SSP5-8.5 climate scenario
```

## 📦 File Format

- All files are in **GeoTIFF (.tif)** format
- Coordinate reference system: **WGS84 (EPSG:4326)**
- Spatial resolution: ~**5 km** (downsampled from original 1 km using mean aggregation)
- Data type: Single-band raster

## 🧪 Variable Categories

- **Topography**: Elevation, Slope, Aspect, TWI
- **Climate**: Precipitation, Temperature, BioClim variables, SPEI
- **Vegetation**: NDVI, NPP, NMDI, NDWI
- **Soil**: Soil moisture, bulk density, sand, clay, silt, pH
- **Human influence**: GLW4 (livestock density), DIS2Roades, DIS2Ditches

## ⚠ Disclaimer

These data are simplified and resampled representations of the full-resolution modeling inputs. They are intended **solely for demonstration and reproduction of code execution**, not for scientific analysis or inference.

## 📄 Citation

If you use this dataset, please cite the corresponding paper:

> X. et al. (2025). *Climate–human interactions drive widespread peatland subsidence and carbon vulnerability in China*. (Submitted)


## 🛠 Contact

For questions or full-resolution data access, please contact:

**Zhenshan Xue**   
Email: [xueshenshan@iga.ac.cn]  
