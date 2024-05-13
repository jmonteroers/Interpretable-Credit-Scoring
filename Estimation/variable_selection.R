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

# Backward Selection
# full_model <- glm(TARGET ~ ., data = train, family = binomial)
# fit_backward <- stepAIC(
#   full_model, direction = "backward", trace = T)

# Forward Selection
null_model <- glm(TARGET ~ 1, data = train, family = binomial)
tic("Forward Selection")
fit_forward <- step(
  null_model, direction = "forward",
  scope=list(lower = ~ 1, upper = formula_full), trace = 1,
  test = "F"
  )
toc()

# Bidirectional Selection
tic("Bidirectional Selection")
null_model <- glm(TARGET ~ 1, data = train, family = binomial)
fit_both <- step(
  null_model, direction = "both", 
  scope=list(lower = ~ 1, upper = formula_full), trace = 1,
  test = "F"
)
toc()

# Extract selected variables from the fitted model object
get_selected_variables <- function(fit) {
  selected_vars <- names(coef(fit))
  # Remove intercept term if present
  selected_vars <- selected_vars[selected_vars != "(Intercept)"]
  return(selected_vars)
}

sel_vars_forward <- get_selected_variables(fit_forward)
sel_vars_both <- get_selected_variables(fit_both)
