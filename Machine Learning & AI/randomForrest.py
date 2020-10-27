import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Next, download the iris dataset from its weblink as follows −

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# assign column names to the dataset

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# dataset 

dataset = pd.read_csv(path, names=headernames)
dataset.head()

# Data Preprocessing

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# training data and testing data −

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train the model 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)

# prediction.
y_pred = classifier.predict(X_test)

# print the results 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)