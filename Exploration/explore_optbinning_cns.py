import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from optbinning import OptimalBinning

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# choose the predictor to discretise and the binary target
variable = "mean radius"
x = df[variable].values
y = data.target

# using the Constraint Programming Solver
optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
optb.fit(x, y)

print(optb.status)
print(optb.splits)

# instantiate binning table
binning_table_builder = optb.binning_table
# build the table
binning_table = binning_table_builder.build()

# we do not need the binning table to get the woe transform
x_transform_woe = optb.transform(x, metric="woe")
x_woe = pd.Series(x_transform_woe).value_counts()

breakpoint()