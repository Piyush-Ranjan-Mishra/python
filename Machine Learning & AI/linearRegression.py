%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# calculate the important values for SLR −

def coef_estimation(x, y):

# number of observations n −

n = np.size(x)

# The mean of x and y vector 

m_x, m_y = np.mean(x), np.mean(y)

# cross-deviation and deviation about x as follows −

SS_xy = np.sum(y*x) - n*m_y*m_x
SS_xx = np.sum(x*x) - n*m_x*m_x

#  regression coefficients i.e. b 

b_1 = SS_xy / SS_xx
b_0 = m_y - b_1*m_x
return(b_0, b_1)

# plot the regression line and predict 

def plot_regression_line(x, y, b):

# plot 

plt.scatter(x, y, color = "m", marker = "o", s = 30)

# predict response vector −

y_pred = b[0] + b[1]*x

# plot the regression line with labels

plt.plot(x, y_pred, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# define main() function for providing dataset and calling the function 

def main():
   x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
   b = coef_estimation(x, y)
   print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
   plot_regression_line(x, y, b)
   
if __name__ == "__main__":
main()


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()


X = diabetes.data[:, np.newaxis, 2]

# training and testing 

X_train = X[:-30]
X_test = X[-30:]


y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]



# train the model
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

# predictions 

y_pred = regr.predict(X_test)

# coefficient like MSE, Variance score etc. as follows −

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

#  plot 

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(