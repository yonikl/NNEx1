import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = fetch_california_housing()

df = pd.DataFrame(data=boston['data'])

df.columns = boston['feature_names']

df['Price'] = boston['target']

corr = df.corr()
corr['Price'].sort_values(ascending=False)

X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = boston.target

scaler = sklearn.preprocessing.StandardScaler()

X = scaler.fit_transform(X)

X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)

alpha = 0.05
m = Y_train.size
theta = np.random.rand(X_train.shape[1])


def gradient_descent(x, y, m, theta, alpha):
    cost_list = []  # to record all cost values to this list
    theta_list = []  # to record all theta_0 and theta_1 values to this list
    prediction_list = []
    run = True
    cost_list.append(1e10)  # we append some large value to the cost list
    i = 0
    while run:
        prediction = np.dot(x, theta)  # predicted y values theta_1*x1+theta_2*x2
        prediction_list.append(prediction)
        error = prediction - y  # compare to y_pred[j] - y_actual[j] in other file
        cost = 1 / (2 * m) * np.dot(error.T, error)  # (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (2 / m) * np.dot(x.T, error))
        # compare to theta = theta - learning_rate*grad in other file
        theta_list.append(theta)
        if np.abs(cost_list[i] - cost_list[i + 1]) < 1e-4:  # checking if the change in cost function is less than 10^(-9)
            run = False
        i += 1
    print(i)
    cost_list.pop(0)  # Remove the large number we added in the beginning
    return prediction_list, cost_list, theta_list


def y_predicted(theta, x):
    return np.dot(x, theta)


def regression_test(x_test, theta):
    return y_predicted(theta, x_test)


def error_function(y_actual, y_predicted):
    error = 0
    for i in range(0, len(y_actual)):  # This is the value of n
        error = error + pow((y_actual[i] - y_predicted[i]), 2)  # This is Y-YP
    # return error/(2*len(y_actual))
    return error / (len(y_actual))


prediction_list, cost_list, theta_list = gradient_descent(X_train, Y_train, m, theta, alpha)

theta = theta_list[-1]
print(theta)
pred_df = pd.DataFrame(
    {
        'Actual Value': Y,
        'Predicted Values': np.dot(X, theta),
    }
)

print(pred_df.head(10))
Y_pred = regression_test(X_test, theta)
error_function(Y_test, Y_pred)

MSE_custom_LR_Model = error_function(Y_test, Y_pred)

lm = LinearRegression()
lm.fit(X_train, Y_train)
y_pred_from_sklearn = lm.predict(X_test)
MSE_sklearn_LR_Model = mean_squared_error(Y_test, y_pred_from_sklearn)

print('Linear Regression (From Scratch)', str(MSE_custom_LR_Model))
print('Linear Regression (SKLEARN)', str(MSE_sklearn_LR_Model))

plt.plot(cost_list)
plt.show()