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
library(dplyr)
library(reshape2)
library(ggplot2)

# TODO: Extend titles
create_heatmap_cor <- function(df, min_cor = NULL) {
  # min_cor is used to keep the variables that have at least one correlation 
  # higher in absolute value than min_cor. if min_cor not NULL, returns
  # the list of variables with high correlation
  
  # Compute correlation matrix
  correlation_matrix <- cor(df, use = "pairwise.complete.obs")
  # Convert correlation matrix to long format for ggplot
  long_cor <- melt(correlation_matrix)
  colnames(long_cor)[3] <- "Correlation"
  
  # ensure same order for both columns
  long_cor$Var2 <- factor(long_cor$Var2, levels = rev(unique(long_cor$Var1)))
  
  # if min_cor is not null, then filter by that absolute value
  if (!is.null(min_cor)) {
    # remove diagonal
    high_cor <- long_cor %>% filter(Var1 != Var2)
    # select vars to keep
    high_cor <- high_cor %>% filter(abs(Correlation) > min_cor)
    sel_vars <- unique(high_cor$Var1)
    # filter long_cor based on this selection
    long_cor <- long_cor %>% filter(Var1 %in% !!sel_vars)
  }
  
  # Plot heatmap using ggplot
  plot <- ggplot(long_cor, aes(Var1, Var2, fill = Correlation)) +
    geom_tile(color = "white") +
    # scale_fill_gradient2(
    #   low = "white", high = "blue",
    #   midpoint = -., limits = c(0,1), name = "Correlation") +
    scale_fill_viridis_b() +
    theme_minimal() +
    labs(title = "Correlation Heatmap",
         x = "Variable 1", y = "Variable 2") +
    theme(axis.text.x =element_blank(), axis.text.y = element_blank())
  
  print(plot)
  
  if (!is.null(min_cor)) return(sel_vars)
}

apply_pca <- function(training, test, vars, thres = .8) {
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

plot_perc_var_exp <- function(pca_estimates, sel_comps = NULL, n_comps = NULL) {
  var_exp <- tidy(
    pca_res$pca_estimates, num = 2, type = "variance"
    )
  # center on cumulative variance
  var_exp <- filter(var_exp, terms == "cumulative percent variance")
  if (!is.null(n_comps)) {
    var_exp <- var_exp %>% filter(component <= n_comps)
  }
  plot <- ggplot(var_exp, aes(component, value)) +
    geom_line(colour = "darkturquoise") +
    theme_minimal() +
    labs(title = "Cumulative Percentage of Variance Explained (%)")
  if (!is.null(sel_comps)) {
    plot <- plot + 
      geom_vline(
        xintercept = sel_comps, linetype="dashed", 
        color = "brown1", size=1.25
        )
  }
  plot
}

# TODO: Extend titles
heatmap_pca_estimates <- function(pca_estimates, thres_abs, n_comps) {
  # Plot heatmap of coefficients of PCA components for terms/features. 
  # - n_comps is the number of components to include, i.e., 1 to n_comp PCs are included
  # - thres_abs is used to filter terms, so that a term has at least a coefficient with an
  # absolute value higher than thres_abs of the absolute value of coefficients (quantile-based) for the
  # number of components selected
  pca_estimates <- tidy(pca_estimates, number = 2)
  
  # only keep first n_comp principal components
  pca_estimates$component_num <- as.numeric(
    gsub("[^0-9]", "", pca_estimates$component)
  )
  pca_estimates <- pca_estimates %>% 
    filter(component_num <= n_comps)
  
  # to simplify chart, only keep terms with a coefficient with an absolute 
  # value higher than 100*thres_abs% of the absolute val of coeffs
  median_abs_coef <- quantile(abs(pca_estimates$value), probs = thres_abs)
  imp_pca_estimates <- pca_estimates %>% 
    filter(abs(value) > median_abs_coef)
  imp_terms <- unique(imp_pca_estimates$terms)
  pca_estimates <- pca_estimates %>%
    filter(terms %in% imp_terms)
  
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

## Correlation for training dataset (only numeric)
# TODO: Repeat analysis after WOE - identify multicolinearity
numeric_training <- select_if(training, is.numeric)
high_cor_vars <- create_heatmap_cor(numeric_training, min_cor = .9)

num_without_housing <- select(
  numeric_training,
  !ends_with("_MODE") & !ends_with("_MEDI") & !ends_with("_AVG")
)
high_cor_vars_without_housing <- create_heatmap_cor(num_without_housing, min_cor = .9)

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
plot_perc_var_exp(pca_res$pca_estimates, sel_comps = 7)
heatmap_pca_estimates(pca_res$pca_estimates, .8, 7)