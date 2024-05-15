# Implement forward and bidirectional variable selection using StepAIC. 
# Input: Dataset after IV selection
# Output: Selected Model for Scorecard Estimation (_step)
# It assumes that the directory has been changes to the path of the input
library(tictoc)

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
tic("AIC, Bidirectional Selection")
null_model <- glm(TARGET ~ 1, data = train, family = binomial)
aic_fit_both <- step(
  null_model, direction = "both", 
  scope=list(lower = ~ 1, upper = formula_full), trace = 1,
  test = "F"
)
toc()

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

sel_vars_aic_both <- get_selected_variables(aic_fit_both)
sel_vars_bic_both <- get_selected_variables(bic_fit_both)

# Export train dataset with selected variables
train_aic <- train[, c("TARGET", sel_vars_aic_both)]
write.csv(train_aic, file=gzfile("train_apps_aic.csv.gz"), row.names = F)

train_bic <- train[, c("TARGET", sel_vars_bic_both)]
write.csv(train_bic, file=gzfile("train_apps_bic.csv.gz"), row.names = F)