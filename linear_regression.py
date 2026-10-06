# importing random to shuffle, numpy for array work, and sklearn for the dataset
import random
import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True) # X = 442 patients and 10 columns, y = diabetes progression score

#shuffling the diabetes data, reproducible b/c of seed.
random.seed(0)
shuffled = list(range(len(y)))
random.shuffle(shuffled)


split = int(0.8 * len(y)) #80/20 train test split
train = shuffled[:split]
test = shuffled[split:]
X_train = X[train] #used shuffle row numbers to pull matching rows from X and y
X_test = X[test]
y_train = y[train]
y_test = y[test]

#standardize data sets using training mean/std
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


def add_intercept(X):
    # Put a column of 1s in front so beta[0] acts as the intercept
    return np.column_stack([np.ones(len(X)), X])

Xb_train = add_intercept(X_train)
Xb_test = add_intercept(X_test)

# solving (X^T X) beta = X^T y which is normal equation to find coefficents
def fit_normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

beta_normal = fit_normal_equation(Xb_train, y_train)

#pairing the feature names with betas
feature_names = ["intercept", "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

for i in range(len(beta_normal)):
    print(feature_names[i], round(beta_normal[i], 2))

print("Intercept:", beta_normal[0].round(2), "| Training mean of y:", y_train.mean().round(2))

y_pred = Xb_test @ beta_normal
print("Predicted:", y_pred[:5].round(1))
print("Actual:   ", y_test[:5])