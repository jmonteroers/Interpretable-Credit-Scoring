from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

from utils.utils import PARENT_DIR, TARGET

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
nn_model.fit(X_train, y_train)

sample_X_train = X_train.sample(n=100)
PartialDependenceDisplay.from_estimator(nn_model, sample_X_train, features=["LTV"], centered=True, kind='both')
plt.show()
breakpoint()