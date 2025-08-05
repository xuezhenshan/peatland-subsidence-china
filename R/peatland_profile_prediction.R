data_dir <- "example_data"
output_dir <- "results"

library(randomForest)
library(terra)
library(dplyr)

# ================== Universal peatland property modeling function ==================
peatland_model <- function(data_csv, variable_name, output_prefix) {
  # Read sample points
  points <- read.csv(data_csv)
  
  # Read environmental variables
  file_paths <- list.files(data_dir, pattern = "tif$", full.names = TRUE)
  env_variables <- rast(file_paths)
  names(env_variables) <- basename(tools::file_path_sans_ext(file_paths))
  
  # Extract environmental values at sample locations
  coords <- cbind(points$Longitude, points$Latitude)
  points_sp <- vect(coords, crs = crs(env_variables), type = "points")
  values <- extract(env_variables, points_sp)
  
  # Create final dataset
  env_data <- as.data.frame(values)
  data <- cbind(points, env_data)
  data <- data[, !(names(data) %in% c("Latitude", "Longitude", "ID", "env_ID"))]
  data <- na.omit(data)
  
  # Split data into training and test sets
  set.seed(123)
  training_samples <- sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)
  train_data <- data[training_samples, ]
  test_data <- data[-training_samples, ]
  
  # Train Random Forest model
  formula_str <- as.formula(paste(variable_name, "~ ."))
  model <- randomForest(formula_str, data = train_data,
                        ntree = 1000, mtry = 4, nodesize = 10)
  
  # Predict and evaluate
  predictions <- predict(model, test_data)
  mse <- mean((predictions - test_data[[variable_name]])^2)
  rmse <- sqrt(mse)
  rsq <- cor(predictions, test_data[[variable_name]])^2
  
  cat(output_prefix, "- MSE:", mse, ", RMSE:", rmse, ", R-squared:", rsq, "\n")
  
  # Export evaluation results
  results <- data.frame(MSE = mse, RMSE = rmse, R_squared = rsq)
  write.csv(results, paste0("G:/CarbonPoolPeatland/result/", output_prefix, "_model_evaluation_results.csv"), row.names = FALSE)
  
  # Export variable importance
  importance_scores <- importance(model)
  write.csv(importance_scores, paste0("G:/CarbonPoolPeatland/result/", output_prefix, "_variable_importance.csv"), row.names = TRUE)
  
  # Predict over entire raster area
  predicted_raster <- predict(env_variables, model)
  writeRaster(predicted_raster, filename = paste0("G:/CarbonPoolPeatland/result/predicted_", output_prefix, ".tif"),
              filetype = "GTiff", overwrite = TRUE)
  
  # Visualization (optional)
  plot(predicted_raster, main = paste("Predicted", variable_name, "using All Variables"))
}

# ================== Call function for each target ==================
peatland_model(file.path(data_dir, "dataB.csv"), "BD", "BD")
peatland_model(file.path(data_dir, "dataT.csv"), "TOC", "TOC")
peatland_model(file.path(data_dir, "dataD.csv"), "Depth", "Depth")
