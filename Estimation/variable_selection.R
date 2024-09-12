# Implement forward and bidirectional variable selection using StepAIC. 
# Input: Dataset after IV selection
# Output: Selected Models for Scorecard Estimation (_aic, _bic)
# It assumes that the directory has been changed to the input path
library(tictoc)
library(car)  # for VIF selection
library(dplyr)

RUN_AIC <- T
RUN_BIC <- T

## Util functions ##
# Extract selected variables from the fitted model object
get_selected_variables <- function(fit) {
  selected_vars <- names(coef(fit))
  # Remove intercept term if present
  selected_vars <- selected_vars[selected_vars != "(Intercept)"]
  return(selected_vars)
}

# Remove positive coefficients, return dataset without columns that produce
# positive coefficients in regression
remove_pos_coeffs <- function(data) {
  cleaned_data <- F
  
  while (!cleaned_data) {
    X_cleaned <- subset(data, select=-TARGET)
    this_model <- glm(TARGET ~ ., data = data, family = binomial)
    coeffs_bic <- coefficients(this_model)[-1]
    idx_pos <- match(TRUE, coeffs_bic > 0)
    
    if (!is.na(idx_pos)) {
      # remove column with positive coefficients
      print(paste("Dropping column", colnames(X_cleaned)[idx_pos]))
      X_cleaned <- X_cleaned[-idx_pos]
      data <- cbind(data$TARGET, X_cleaned)
      colnames(data)[1] <- "TARGET"
    } else {
      cleaned_data <- T
    }
  }
  
  list(data = data, last.fit = this_model)
}

variable_selection <- function(train, k, outname){
  # Construct the formula for full model (for upper scope)
  indep_vars <- colnames(train)[2:ncol(train)]
  formula_full <- as.formula(
    paste("~", paste(indep_vars, collapse = " + ")))
  
  # Variable Selection - both directions
  null_model <- glm(TARGET ~ 1, data = train, family = binomial)
  fit_both <- step(
    null_model, direction = "both", 
    scope=list(lower = ~ 1, upper = formula_full), trace = 1, k = k
    )
  sel_vars_both <- get_selected_variables(fit_both)
  sel_train <- train[, c("TARGET", sel_vars_both)]
  sel_model <- glm(TARGET ~ ., data = sel_train, family = binomial)
  
  # VIF Selection
  vif <- vif(sel_model)
  attrs_large_vif <- names(vif[vif > 10.])
  sel_train <- sel_train %>% select(-all_of(attrs_large_vif))
  
  # Removal, positive coefficients (in a loop)
  sel_train <- remove_pos_coeffs(sel_train)
  print(paste(
    "Number of rows in resulting scorecard",
    get_nrows_scorecard(sel_train$data)
  ))
  
  # Output
  write.csv(
    sel_train$data, file=gzfile(outname), row.names = F
  )
}

# to find number of rows of resulting scorecard
get_nrows_scorecard <- function(df) {
  nunique_per_col <- apply(df, 2, function(c) length(unique(c)))
  # substract number of target values
  sum(nunique_per_col) - length(unique(df$TARGET))
}

## Load dataset ##
train <- read.csv(unz("train_apps_iv.csv.zip", "train_apps_iv.csv"))
train <- subset(train, select=-SK_ID_CURR)

## AIC - Bidirectional Selection ##
if (RUN_AIC) {
  tic("AIC, Bidirectional Selection")
  variable_selection(train, 2., "train_apps_aic.csv.gz")
  toc()
}

## BIC - Bidirectional Selection ##
if (RUN_BIC) {
  k_bic <- log(nrow(train))
  tic("BIC, Bidirectional Selection")
  variable_selection(train, k_bic, "train_apps_bic.csv.gz")
  toc()
}