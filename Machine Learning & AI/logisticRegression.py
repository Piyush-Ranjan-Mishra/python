Import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# load dataset
digits = datasets.load_digits()

# the feature matrix(X) and response vector(y)
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# object of logistic regression 

digreg = linear_model.LogisticRegression()

#  train the model by using the training sets as follows −

digreg.fit(X_train, y_train)
print the accuracy of the model as follows −

print("Accuracy of Logistic Regression model is:",
metrics.accuracy_score(y_test, y_pred)*100)
