# This script processes the features by semantic groups. For each semantic group,
# produces visual illustrations to study multicorrelation. If detected, carries
# out a PCA in order to reduce the dimensionality

# - Input: train/test_apps_imp.csv.zip
# - Output: train/test_apps_reduced.csv.zip

# Currently, analyses the semantic group(s): Housing

# Assumes that the working directory has been set to where the input datasets are
# Saves output in the same folder

library(recipes)
library(tidyr)
library(reshape2)
library(ggplot2)

# TODO: Extend titles
create_heatmap_cor <- function(df) {
  # Compute correlation matrix
  correlation_matrix <- cor(df, use = "pairwise.complete.obs")
  # Convert correlation matrix to long format for ggplot
  long_cor <- melt(correlation_matrix)
  colnames(long_cor)[3] <- "Correlation"
  
  # ensure same order for both columns
  long_cor$Var2 <- factor(long_cor$Var2, levels = rev(unique(long_cor$Var1)))
  
  # Plot heatmap using ggplot
  plot <- ggplot(long_cor, aes(Var1, Var2, fill = Correlation)) +
    geom_tile(color = "white") +
    theme_minimal() +
    labs(title = "Correlation Heatmap") +
    theme(axis.text.x =element_blank(), axis.text.y = element_blank())
  
  plot
}

apply_pca <- function(training, test, vars, thres = .8) {
  # Apply PCA using ... as filters
  # Note: imputing missing values using mean to avoid losing much information
  pca_rec <- recipe(TARGET ~ ., data = training) %>%
    step_impute_mean(!!vars) %>%
    step_pca(
      !!vars,
      threshold = thres,
      options=list(center = T, scale. = T))
  
  # Prepare and bake the datasets
  pca_estimates <- prep(pca_rec, training = training)
  training <- bake(pca_estimates, new_data = training)
  testing <- bake(pca_estimates, new_data = testing)
  
  list(training = training, testing = testing, pca_estimates = pca_estimates)
}

# TODO: Extend titles
heatmap_pca_estimates <- function(pca_estimates, thres_abs, n_comps) {
  pca_estimates <- tidy(pca_estimates, number = 2)
  # to simplify chart, only keep coefficients that are larger than
  # quantile threshold for absolute value
  median_abs_coef <- quantile(abs(pca_estimates$value), probs = thres_abs)
  # only keep first n_comp principal components
  pca_estimates$component_num <- as.numeric(
    gsub("[^0-9]", "", pca_estimates$component)
  )
  pca_estimates <- pca_estimates %>% 
    filter(abs(value) > median_abs_coef, component_num <= n_comps)
  # Plot heatmap using ggplot
  plot <- ggplot(pca_estimates, aes(x = component, y = terms, fill = value)) +
    geom_tile(colour = "white") +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red", 
      midpoint = 0, limits = c(-1,1), name = "Coefficient") +
    theme_minimal() +
    labs(title = "PCA Heatmap") +
    theme(panel.grid.major = element_blank())
  
  plot
}

# Load data
training <- read.csv(gzfile("train_apps_imp.csv.gz"))
testing <- read.csv(gzfile("test_apps_imp.csv.gz"))

## PCA for Housing Variables

# Exploratory Heatmap
# Select data related to the numerical columns for housing
housing <- select(
  training,
  where(is.numeric) & 
  (ends_with("_MODE") | ends_with("_MEDI") | ends_with("_AVG"))
)
# note: subdiagonals, different levels of positive correlation
create_heatmap_cor(housing)

# Apply PCA
vars_pca_housing <- names(
  training %>% select(
    where(is.numeric) & matches(".+_((MODE)|(MEDI)|(AVG))$"))
  )
# Note: using na.omit in pca step only relies on 50k observations
# so imputing mean before applying PCA
pca_res <- apply_pca(
  training, test, vars_pca_housing
  )

# Analyse coefficients of PCA for Housing
heatmap_pca_estimates(pca_res$pca_estimates, .5, 7)
