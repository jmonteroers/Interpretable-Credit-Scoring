# Implement forward and bidirectional variable selection using StepAIC. 
# Input: Dataset after IV selection
# Output: Selected Model for Scorecard Estimation (_step)
# It assumes that the directory has been changes to the path of the input
library(tictoc)
library(car)  # for VIF selection

setwd("C:/Users/jmont/Documents/courses/msc-data-science/master-thesis/credit-scoring/TFM/Data/Home Credit/processed")

# Load dataset
train <- read.csv(unz("train_apps_iv.csv.zip", "train_apps_iv.csv"))
train <- subset(train, select=-SK_ID_CURR)

# Construct the formula for full model (for upper scope)
indep_vars <- colnames(train)[2:ncol(train)]
formula_full <- as.formula(
  paste("~", paste(indep_vars, collapse = " + ")))

# AIC

# Backward Selection
# full_model <- glm(TARGET ~ ., data = train, family = binomial)
# fit_backward <- stepAIC(
#   full_model, direction = "backward", trace = T)

# Forward Selection
# null_model <- glm(TARGET ~ 1, data = train, family = binomial)
# tic("Forward Selection")
# fit_forward <- step(
#   null_model, direction = "forward",
#   scope=list(lower = ~ 1, upper = formula_full), trace = 1,
#   test = "F"
#   )
# toc()

# Bidirectional Selection
# tic("AIC, Bidirectional Selection")
# null_model <- glm(TARGET ~ 1, data = train, family = binomial)
# aic_fit_both <- step(
#   null_model, direction = "both", 
#   scope=list(lower = ~ 1, upper = formula_full), trace = 1,
#   test = "F"
# )
# toc()

# BIC
k_bic <- log(nrow(train))
tic("BIC, Bidirectional Selection")
null_model <- glm(TARGET ~ 1, data = train, family = binomial)
bic_fit_both <- step(
  null_model, direction = "both", 
  scope=list(lower = ~ 1, upper = formula_full), trace = 1,
  test = "F", k = k_bic
)
toc()

# Extract selected variables from the fitted model object
get_selected_variables <- function(fit) {
  selected_vars <- names(coef(fit))
  # Remove intercept term if present
  selected_vars <- selected_vars[selected_vars != "(Intercept)"]
  return(selected_vars)
}

# sel_vars_aic_both <- get_selected_variables(aic_fit_both)
sel_vars_bic_both <- get_selected_variables(bic_fit_both)

# Export train dataset with selected variables
# train_aic <- train[, c("TARGET", sel_vars_aic_both)]
# write.csv(train_aic, file=gzfile("train_apps_aic.csv.gz"), row.names = F)

train_bic <- train[, c("TARGET", sel_vars_bic_both)]
write.csv(train_bic, file=gzfile("train_apps_bic.csv.gz"), row.names = F)

# VIC selection

# to check number of rows of resulting scorecard
get_nrows_scorecard <- function(df) {
  nunique_per_col <- apply(train_bic, 2, function(c) length(unique(c)))
  # substract number of target values
  sum(nunique_per_col) - length(unique(df$TARGET))
}

# BIC
train_bic <- read.csv(gzfile("train_apps_bic.csv.gz"))
bic_model <- glm(TARGET ~ ., data = train_bic, family = binomial)
# No VIF issues
vif_bic <- vif(bic_model)

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
      data <- cbind(train_bic$TARGET, X_cleaned)
      colnames(data)[1] <- "TARGET"
    } else {
      cleaned_data <- T
    }
  }
  
  list(data = data, last.fit = this_model)
}

cleaned_train_bic <- remove_pos_coeffs(train_bic)
write.csv(
  cleaned_train_bic$data, file=gzfile("train_apps_bic_npos.csv.gz"), row.names = F
  )
print(paste(
  "Number of rows in resulting scorecard",
  get_nrows_scorecard(cleaned_train_bic)
  )
)