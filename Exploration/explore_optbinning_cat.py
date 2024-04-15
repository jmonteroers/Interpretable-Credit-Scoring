import pandas as pd

from optbinning import OptimalBinning

df_cat = pd.read_csv(r"C:\Users\jmont\Documents\courses\msc-data-science\master-thesis\credit-scoring\TFM\Data\Home Credit\application_train.csv",
                     engine='c')


variable_cat = "NAME_INCOME_TYPE"
x_cat = df_cat[variable_cat].values
y_cat = df_cat.TARGET.values

print(df_cat[variable_cat].value_counts())

optb = OptimalBinning(name=variable_cat, dtype="categorical", solver="mip",
                      cat_cutoff=0.1)

optb.fit(x_cat, y_cat)

print(optb.status)

binning_table = optb.binning_table
print(binning_table.build())


x_new = ["Businessman", "Working", "New category"]
x_transform_woe = optb.transform(x_new, metric="woe")
print(pd.DataFrame({variable_cat: x_new, "WoE": x_transform_woe}))
breakpoint()