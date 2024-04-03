# Script that imputes the missing values for the columns with no clear bivariate relation
# - Input: train/test_apps_ext.csv.zip
# - Output: train/test_apps_imp.csv.zip

# Variables to be imputed (following the bivariate missing report)
# EXT_SOURCE_2, AMT_GOODS_PRICE, AMT_ANNUITY, CNT_FAM_MEMBERS, 
# DAYS_LAST_PHONE_CHANGE
# Dependent variables that are recreated
# LB_Credit_Length and LTV

# Assumes that the working directory has been set to where the input datasets are
# Saves output in same folder

library(recipes)

# Load data
training <- read.csv(unz("train_apps_ext.csv.zip", "train_apps_ext.csv"))
testing <- read.csv(unz("test_apps_ext.csv.zip", "test_apps_ext.csv"))

# TEMP: for debugging purposes
# aux_training <- training[is.na(training$EXT_SOURCE_2), ]
# aux_training <- rbind(aux_training, training[1:10000, ])
# training <- aux_training
# gc()

# Initial checks - are there any missing vals?
target_columns <- c(
  "EXT_SOURCE_2", "AMT_GOODS_PRICE", "AMT_ANNUITY", "CNT_FAM_MEMBERS", 
  "DAYS_LAST_PHONE_CHANGE"
)
print("Before")
print(summary(training[, target_columns]))
print(summary(testing[, target_columns]))

# drop LTV and LB_Credit_Length since they will be recreated
# to avoid using them as predictors (redundant information)
training <- training[, !(colnames(training) %in% c("X", "LB_Credit_Length", "LTV"))]


# Build recipe, add knn step with 5 neighbors
# not consider the id as a predictor!
knn_rec <- recipe(TARGET ~ ., data = training) %>%
    step_string2factor(all_nominal_predictors()) %>%
    step_impute_knn(
        EXT_SOURCE_2, AMT_GOODS_PRICE, AMT_ANNUITY, CNT_FAM_MEMBERS, 
        DAYS_LAST_PHONE_CHANGE,
        impute_with = imp_vars(all_predictors(), -SK_ID_CURR),
        neighbors = 5
    )

# Prepare and bake the datasets
trained_knn_rec <- prep(knn_rec, training = training)
training <- bake(trained_knn_rec, new_data = training)
testing <- bake(trained_knn_rec, new_data = training)

# Recreate LTV and LB_Credit_Length
training["LTV"] <- training$AMT_CREDIT / training$AMT_GOODS_PRICE
training["LB_Credit_Length"] <- training$AMT_CREDIT / training$AMT_ANNUITY

# Final checks - are there any missing vals?
print("After")
print(summary(training[, target_columns]))
print(summary(testing[, target_columns]))

# Save results
write.csv(training, file=gzfile("train_apps_imp.csv.gz"))
write.csv(testing, file=gzfile("test_apps_imp.csv.gz"))