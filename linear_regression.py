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
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#standardize training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


def add_intercept(X):
    # Put a column of 1s in front so beta[0] acts as the intercept
    return np.column_stack([np.ones(len(X)), X])

Xb_train = add_intercept(X_train)
Xb_test = add_intercept(X_test)

print(X_train.mean(axis=0).round(3))
print(X_train.std(axis=0).round(3))
print(Xb_train.shape, Xb_test.shape)


