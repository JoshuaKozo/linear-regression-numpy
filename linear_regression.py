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

# Gradient descent where we start at coefficents = zero and figure out best way to improve error using gradient
def fit_gradient_descent(X, y, learning_rate=0.1, n_iters=5000):
    n = len(y)
    beta = np.zeros(X.shape[1])
    losses = []
    for i in range(n_iters):
        errors = X @ beta - y
        losses.append(np.mean(errors ** 2))
        gradient = (2 / n) * X.T @ errors
        beta = beta - learning_rate * gradient
    return beta, losses

beta_gd, losses = fit_gradient_descent(Xb_train, y_train)

print("Starting MSE:", round(losses[0], 1))
print("Final MSE:   ", round(losses[-1], 1))
print("Biggest difference from normal equation:", round(np.max(np.abs(beta_gd - beta_normal)), 4))


#now lets check that the models are actually working with stikit-learn's linear regression
from sklearn.linear_model import LinearRegression

sk_model = LinearRegression().fit(X_train, y_train)
beta_sk = np.concatenate([[sk_model.intercept_], sk_model.coef_])

#comparison of methods
methods = ["Normal equation", "Gradient descent", "scikit-learn"]
betas = [beta_normal, beta_gd, beta_sk]

# mean squared error calculation
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# R^2 calculation
def r_squared(y, y_pred):
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)


for i in range(len(methods)):
    y_pred = Xb_test @ betas[i]
    print(methods[i], "| test MSE:", round(mse(y_test, y_pred), 1), "| test R^2:", round(r_squared(y_test, y_pred), 3))

print("Normal equation vs scikit-learn:", np.max(np.abs(beta_normal - beta_sk)))
print("Gradient descent vs scikit-learn:", np.max(np.abs(beta_gd - beta_sk)))