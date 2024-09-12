from sklearn.neural_network import MLPClassifier
import pandas as pd
from alibi.explainers import CounterfactualProto
import tensorflow as tf
    
from utils.utils import PARENT_DIR, TARGET

# to avoid errors related to eager execution
tf.compat.v1.disable_eager_execution()
RANDOM_SEED = 1234

# Load data, split into X, y
train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")
    
X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
y_train = train[TARGET]

nn_model = MLPClassifier(
    learning_rate="invscaling", learning_rate_init=0.001, 
    max_iter=100, early_stopping=True, random_state=RANDOM_SEED
    )
nn_model.fit(X_train.values, y_train)

shape = (1,) + X_train.shape[1:]
predict_fn = lambda x: nn_model.predict_proba(x)
breakpoint()
cf = CounterfactualProto(predict_fn, shape, use_kdtree=True, theta=10., feature_range=(-1., 1.))
cf.fit(X_train.values)
# Retrieve first default in test
default_obs = test.loc[test.TARGET == 1].iloc[0:1]
default_obs = default_obs.drop(columns=[TARGET, "SK_ID_CURR"]).values
# ISSUE: explanation must be much sparser
explanation = cf.explain(default_obs, k=10)

breakpoint()