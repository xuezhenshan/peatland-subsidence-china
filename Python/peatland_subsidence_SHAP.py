import os
data_dir = os.path.join(os.path.dirname(__file__), "example_data")
output_dir = os.path.join(os.path.dirname(__file__), "output")

import os
import numpy as np
import pandas as pd
import rasterio
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import rcParams

# ===== Path Setup =====
point_csv = ros.path.join(data_dir, "Sampling.csv")
base_folder = output_dir
current_folder = os.path.join(base_folder, "Current")
result_dir = os.path.join(base_folder, "Result_AllVariable_SHAP_Finnal1")
os.makedirs(result_dir, exist_ok=True)

# ===== Load Data =====
df = pd.read_csv(point_csv)
all_vars = sorted([f for f in os.listdir(current_folder) if f.endswith('.tif') and f != "Elevation.tif"])

# ===== Feature Extraction =====
def extract_features(folder, points_df, variables):
    features = []
    for var in variables:
        with rasterio.open(os.path.join(folder, var)) as src:
            coords = [(x, y) for x, y in zip(points_df["X"], points_df["Y"])]
            values = [val[0] for val in src.sample(coords)]
            features.append(values)
    return pd.DataFrame(np.array(features).T, columns=variables)

X_all = extract_features(current_folder, df, all_vars)
X_all["Value"] = df["Value"]
X_all.dropna(inplace=True)
X = X_all[all_vars]
y = X_all["Value"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Train Model =====
model = XGBRegressor(n_estimators=1500, learning_rate=0.01, max_depth=10,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
                     random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# ===== SHAP Analysis =====
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap_df = pd.DataFrame(shap_values, columns=all_vars)

mean_shap = shap_df.abs().mean()
std_shap = shap_df.abs().std()
shap_summary = pd.DataFrame({
    "Variable": mean_shap.index,
    "Mean_ABS_SHAP": mean_shap.values,
    "Std_ABS_SHAP": std_shap[mean_shap.index].values
})
shap_summary.to_csv(os.path.join(result_dir, "AllVariable_SHAP_Summary_MeanStd.csv"), index=False)

# ===== Feature Importance =====
importance_df = pd.DataFrame({
    "Variable": all_vars,
    "Importance": model.feature_importances_
})
importance_df.to_csv(os.path.join(result_dir, "AllVariableImportance.csv"), index=False)

# ===== Variable Group Mapping (correct order, Soil first) =====
group_map = {}
for v in all_vars:
    if  v in ["Tem.tif", "Pre.tif", "Swe.tif", "SPEI.tif", "Scd.tif", "SHEI.tif"] or v.startswith("Bio"):
        group_map[v] = "Climate"
    elif v in ["GLW4.tif", "GroundWater.tif", "DIS2Roades.tif", "DIS2Ditches.tif"]:
        group_map[v] = "Human influence"
    elif v in ["Aspect.tif", "Slope.tif", "Twi.tif"]:
        group_map[v] = "Topography"
    elif v in ["NDVI.tif", "NDWI.tif", "NPP.tif", "NMDI.tif"]:
        group_map[v] = "Vegetation"
    else:
        group_map[v] = "Soil"

shap_summary["Group"] = shap_summary["Variable"].map(group_map)
importance_df["Group"] = importance_df["Variable"].map(group_map)

shap_summary["SortOrder"] = shap_summary.groupby("Group")["Mean_ABS_SHAP"].rank("first", ascending=False)
shap_summary["GroupOrder"] = shap_summary["Group"].map({
    "Climate": 0, "Soil": 1, "Human influence": 2, "Topography": 3, "Vegetation": 4, "Other": 5
})
shap_summary = shap_summary.sort_values(["GroupOrder", "SortOrder"])
ordered_vars = shap_summary["Variable"].tolist()

# ===== Font Config =====
rcParams['font.family'] = 'Times New Roman'

# ===== Violin Plot =====
shap_plot_df = shap_df[ordered_vars].melt(var_name="Variable", value_name="SHAP Value")
shap_plot_df["Group"] = shap_plot_df["Variable"].map(group_map)

plt.figure(figsize=(12, 14))
sns.violinplot(data=shap_plot_df, y="Variable", x="SHAP Value", hue="Group",
               palette="Set2", scale="width", linewidth=1, inner="quartile", dodge=False)
plt.title("SHAP Value Distribution by Variable Group", fontsize=16)
plt.xlabel("SHAP Value", fontsize=14)
plt.ylabel("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "SHAP_Violin_Grouped.svg"), dpi=600)
plt.close()

# ===== Color Mapping for Barplots =====
group_palette = sns.color_palette("Set2", 5)
group_order = ["Climate", "Soil", "Human influence", "Topography", "Vegetation"]
group_color_map = dict(zip(group_order, group_palette))

shap_sorted = shap_summary.set_index("Variable").loc[ordered_vars]
importance_sorted = importance_df.set_index("Variable").loc[ordered_vars]
shap_colors = shap_sorted["Group"].map(group_color_map)
importance_colors = importance_sorted["Group"].map(group_color_map)

# ===== SHAP Barplot =====
plt.figure(figsize=(10, 14))
plt.barh(shap_sorted.index, shap_sorted["Mean_ABS_SHAP"],
         xerr=shap_sorted["Std_ABS_SHAP"], color=shap_colors, edgecolor="black", capsize=3)
plt.xlabel("Mean Absolute SHAP Value", fontsize=13)
plt.title("SHAP Importance (All Variables)", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "SHAP_Importance_All_ColorMatched.svg"), dpi=600)
plt.close()

# ===== Feature Importance Barplot =====
plt.figure(figsize=(10, 14))
plt.barh(importance_sorted.index, importance_sorted["Importance"],
         color=importance_colors, edgecolor="black")
plt.xlabel("XGBoost Feature Importance", fontsize=13)
plt.title("Feature Importance (All Variables)", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "FeatureImportance_All_ColorMatched.svg"), dpi=600)
plt.close()

# ===== Prediction: Current Raster =====
with rasterio.open(os.path.join(current_folder, all_vars[0])) as ref:
    width, height = ref.width, ref.height
    transform = ref.transform
    crs = ref.crs
profile = {
    'driver': 'GTiff', 'height': height, 'width': width,
    'count': 1, 'dtype': 'float32', 'crs': crs, 'transform': transform, 'compress': 'lzw'
}
stack = []
for var in all_vars:
    with rasterio.open(os.path.join(current_folder, var)) as src:
        stack.append(src.read(1))
data = np.stack(stack, axis=-1).reshape(-1, len(all_vars))
mask = ~np.any(np.isnan(data), axis=1)
pred = np.full(data.shape[0], np.nan, dtype=np.float32)
pred[mask] = model.predict(data[mask])
pred = pred.reshape(height, width)
with rasterio.open(os.path.join(result_dir, "Prediction_Current_AllVariable.tif"), 'w', **profile) as dst:
    dst.write(pred, 1)

# ===== Prediction: Future Scenarios =====
scenario_folders = {
    "SSP126": os.path.join(base_folder, "2100_SSP_126"),
    "SSP370": os.path.join(base_folder, "2100_SSP_370"),
    "SSP585": os.path.join(base_folder, "2100_SSP_585")
}
for scen, scen_folder in scenario_folders.items():
    with rasterio.open(os.path.join(scen_folder, all_vars[0])) as ref:
        width, height = ref.width, ref.height
        transform = ref.transform
        crs = ref.crs
    profile = {
        'driver': 'GTiff', 'height': height, 'width': width,
        'count': 1, 'dtype': 'float32', 'crs': crs, 'transform': transform, 'compress': 'lzw'
    }
    stack = []
    for var in all_vars:
        with rasterio.open(os.path.join(scen_folder, var)) as src:
            stack.append(src.read(1))
    data = np.stack(stack, axis=-1).reshape(-1, len(all_vars))
    mask = ~np.any(np.isnan(data), axis=1)
    pred = np.full(data.shape[0], np.nan, dtype=np.float32)
    pred[mask] = model.predict(data[mask])
    pred = pred.reshape(height, width)
    out_tif = os.path.join(result_dir, f"Prediction_{scen}_AllVariable.tif")
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(pred, 1)

# ========== Output Model Performance ==========
performance_df = pd.DataFrame({
    "R_squared": [r2],
    "RMSE": [rmse]
})
performance_df.to_csv(os.path.join(result_dir, "Model_Performance.csv"), index=False)

# Save SHAP summary with group and color
shap_summary[["Variable", "Mean_ABS_SHAP", "Std_ABS_SHAP", "Group", "Color"]].to_csv(
    os.path.join(result_dir, "AllVariable_SHAP_Summary_MeanStd_Grouped.csv"), index=False
)

# Save feature importance (ordered same as SHAP)
importance_df.to_csv(os.path.join(result_dir, "AllVariable_Importance_Ordered.csv"), index=False)
