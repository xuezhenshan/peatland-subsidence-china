data_dir <- "example_data"
output_dir <- "results"

# ================== Step 0. Set working directory ==================
output_dir <- "results"

# ================== Step 1. Read sampling data and environmental predictors ==================
peatland_data <- read.csv(file.path(work_dir, "Peatland_Sampling.csv"))
peatland_data$x <- as.numeric(peatland_data$x)
peatland_data$y <- as.numeric(peatland_data$y)
peatland_data <- peatland_data[!is.na(peatland_data$x) & !is.na(peatland_data$y), ]
if(nrow(peatland_data) == 0) stop("Error: No valid sampling points after removing NA coordinates.")

env_dir <- file.path(work_dir, "current")
if(!dir.exists(env_dir)) stop(paste0("Error: Folder not found: ", env_dir))
env_files <- list.files(env_dir, pattern = "tif$", full.names = TRUE)
if(length(env_files) == 0) stop(paste0("Error: No tif files found in ", env_dir))
env_rasters <- rast(env_files)

points_sf <- st_as_sf(peatland_data, coords = c("x", "y"), crs = crs(env_rasters))
env_values <- terra::extract(env_rasters, vect(points_sf))
peatland_data <- cbind(peatland_data, env_values[,-1])

# ================== Step 2. Generate pseudo-absence points ==================
set.seed(42)
n_background <- 100000
background_points <- spatSample(env_rasters, size = n_background, method = "random", as.points = TRUE)
background_values <- terra::extract(env_rasters, background_points)
coords <- crds(background_points)
background_data <- as.data.frame(background_values[,-1])
background_data$x <- coords[,1]
background_data$y <- coords[,2]
background_data$Presence <- 0

peatland_data$Presence <- 1
training_data <- rbind(
  peatland_data[, c("x", "y", names(env_rasters), "Presence")],
  background_data
)
training_data <- na.omit(training_data)

# ================== Step 3. Split dataset (70% training, 30% testing) ==================
set.seed(42)
train_index <- sample(1:nrow(training_data), 0.7 * nrow(training_data))
train_set <- training_data[train_index, ]
test_set <- training_data[-train_index, ]
predictors <- names(env_rasters)

# ================== Step 4. Model training ==================
rf_model <- randomForest(as.factor(Presence) ~ ., data = train_set[, c(predictors, "Presence")], ntree = 500)
gbm_model <- gbm(Presence ~ ., data = train_set[, c(predictors, "Presence")],
                 distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
train_matrix <- xgb.DMatrix(data = as.matrix(train_set[, predictors]), label = train_set$Presence)
xgb_model <- xgboost(data = train_matrix, nrounds = 100, objective = "binary:logistic", verbose = 0)

# ================== Step 5. Extract variable importance ==================
rf_importance <- importance(rf_model)
gbm_importance <- summary.gbm(gbm_model, plotit = FALSE)
xgb_importance <- xgb.importance(model = xgb_model)

rf_df <- data.frame(Variable = rownames(rf_importance), RF_Importance = rf_importance[,1])
gbm_df <- data.frame(Variable = gbm_importance$var, GBM_Importance = gbm_importance$rel.inf)
xgb_df <- data.frame(Variable = xgb_importance$Feature, XGB_Importance = xgb_importance$Gain)

importance_all <- full_join(rf_df, gbm_df, by = "Variable") %>%
  full_join(xgb_df, by = "Variable")
write.csv(importance_all, file.path(work_dir, "Variable_Importance_Scores_all_models.csv"), row.names = FALSE)

# ================== Step 6. Model evaluation (internal validation) ==================
test_matrix <- xgb.DMatrix(data = as.matrix(test_set[, predictors]))
rf_pred <- predict(rf_model, test_set[, predictors], type = "prob")
if (is.matrix(rf_pred)) { rf_pred <- rf_pred[,2] } else { rf_pred <- as.numeric(rf_pred) }
gbm_pred <- predict(gbm_model, test_set[, predictors], n.trees = 100, type = "response")
xgb_pred <- predict(xgb_model, test_matrix)
ensemble_pred <- (rf_pred + gbm_pred + xgb_pred) / 3

evaluate_model <- function(obs, pred) {
  roc_obj <- roc(obs, pred)
  auc_val <- auc(roc_obj)
  pred_binary <- ifelse(pred >= 0.5, 1, 0)
  cm <- table(factor(pred_binary, levels = c(0,1)), factor(obs, levels = c(0,1)))
  sensitivity <- ifelse(sum(cm[,2]) == 0, 0, cm[2,2] / sum(cm[,2]))
  specificity <- ifelse(sum(cm[,1]) == 0, 0, cm[1,1] / sum(cm[,1]))
  tss_val <- sensitivity + specificity - 1
  TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]; TP <- cm[2,2]
  total <- TN + FP + FN + TP
  po <- (TP + TN) / total
  pe <- ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)) / (total^2)
  kappa_val <- ifelse(1 - pe == 0, NA, (po - pe) / (1 - pe))
  return(c(AUC = auc_val, TSS = tss_val, Kappa = kappa_val))
}

rf_metrics <- evaluate_model(test_set$Presence, rf_pred)
gbm_metrics <- evaluate_model(test_set$Presence, gbm_pred)
xgb_metrics <- evaluate_model(test_set$Presence, xgb_pred)
ensemble_metrics <- evaluate_model(test_set$Presence, ensemble_pred)

internal_results <- data.frame(
  Model = c("RandomForest", "GBM", "XGBoost", "Ensemble"),
  AUC = c(rf_metrics["AUC"], gbm_metrics["AUC"], xgb_metrics["AUC"], ensemble_metrics["AUC"]),
  TSS = c(rf_metrics["TSS"], gbm_metrics["TSS"], xgb_metrics["TSS"], ensemble_metrics["TSS"]),
  Kappa = c(rf_metrics["Kappa"], gbm_metrics["Kappa"], xgb_metrics["Kappa"], ensemble_metrics["Kappa"])
)

# ================== Step 7. Raster prediction ==================
env_stack <- as.data.frame(values(env_rasters))
colnames(env_stack) <- names(env_rasters)
rf_full <- predict(rf_model, env_stack, type = "prob")
if (is.matrix(rf_full)) { rf_full <- rf_full[,2] } else { rf_full <- as.numeric(rf_full) }
gbm_full <- predict(gbm_model, env_stack, n.trees = 100, type = "response")
xgb_full <- predict(xgb_model, newdata = as.matrix(env_stack))
ensemble_full <- (rf_full + gbm_full + xgb_full) / 3

pred_raster <- env_rasters[[1]]
pred_raster[] <- ensemble_full
writeRaster(pred_raster, filename = file.path(work_dir, "Predicted_Ensemble_Suitability.tif"), overwrite = TRUE)

binary_raster <- pred_raster
binary_raster[] <- ifelse(ensemble_full >= 0.5, 1, 0)
writeRaster(binary_raster, filename = file.path(work_dir, "Predicted_Ensemble_Binary.tif"), overwrite = TRUE)

# ================== Step 8. External validation ==================
test_points <- vect(cbind(test_set$x, test_set$y), crs = crs(pred_raster))
predicted_values <- extract(binary_raster, test_points)[,2]
cm <- table(factor(predicted_values, levels = c(0,1)), factor(test_set$Presence, levels = c(0,1)))
TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]; TP <- cm[2,2]
total <- TN + FP + FN + TP
po <- (TP + TN) / total
pe <- ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)) / (total^2)
external_kappa_val <- ifelse(1 - pe == 0, NA, (po - pe) / (1 - pe))

internal_results$ExternalValidationKappa <- NA
internal_results$ExternalValidationKappa[internal_results$Model == "Ensemble"] <- external_kappa_val
write.csv(internal_results, file.path(work_dir, "Final_Model_Performance_Results_with_ExternalKappa.csv"), row.names = FALSE)