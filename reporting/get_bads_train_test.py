from utils import load_train_test


train, X_train, y_train, test, X_test, y_test = load_train_test()

# Train
print(f"Train, N: {train.shape[0]}")
print(f"Train, N Bads: {y_train.sum()}")
print(f"Train, % Bads: {round(100*y_train.mean(), 2)}")

# Test
print(f"Test, N: {test.shape[0]}")
print(f"Test, N Bads: {y_test.sum()}")
print(f"Test, % Bads: {round(100*y_test.mean(), 2)}")

# Total
print(f"Total, N: {train.shape[0] + test.shape[0]}")
print(f"Total, N Bads: {y_train.sum() + y_test.sum()}")
prop_bads_total = (y_train.sum() + y_test.sum()) / (train.shape[0] + test.shape[0])
print(f"Total, % Bads: {round(100*prop_bads_total, 2)}")