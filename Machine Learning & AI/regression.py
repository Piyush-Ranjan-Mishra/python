import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# own dataset to build regression prediction model

input = r'C:\linear.txt'

# np.loadtxt function to load it.

input_data = np.loadtxt(input, delimiter=',')
X, y = input_data[:, :-1], input_data[:, -1]

# training set and a test set

training_samples = int(0.6 * len(X))
testing_samples = len(X) - num_training

X_train, y_train = X[:training_samples], y[:training_samples]

X_test, y_test = X[training_samples:], y[training_samples:]


# a linear regressor object.

reg_linear= linear_model.LinearRegression()

# train 

reg_linear.fit(X_train, y_train)

# prediction with the testing data.

y_test_pred = reg_linear.predict(X_test)

# Plot & visualization

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_pred, color='black', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()