import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# assign column names to the dataset as follows −

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


dataset = pd.read_csv(path, names=headernames)
dataset.head()

# slno. 	sepal-length 	sepal-width 	petal-length 	petal-width 	Class
# 0 	5.1 	3.5 	1.4 	0.2 	Iris-setosa
# 1 	4.9 	3.0 	1.4 	0.2 	Iris-setosa
# 2 	4.7 	3.2 	1.3 	0.2 	Iris-setosa
# 3 	4.6 	3.1 	1.5 	0.2 	Iris-setosa
# 4 	5.0 	3.6 	1.4 	0.2 	Iris-setosa

# Data Preprocessing 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

# data scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train the model 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

# Predictions

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

# Output

# Confusion Matrix:
# [[21 0 0]
# [ 0 16 0]
# [ 0 7 16]]
# Classification Report:
#             precision      recall       f1-score       support
# Iris-setosa       1.00        1.00         1.00          21
# Iris-versicolor   0.70        1.00         0.82          16
# Iris-virginica    1.00        0.70         0.82          23
# micro avg         0.88        0.88         0.88          60
# macro avg         0.90        0.90         0.88          60
# weighted avg      0.92        0.88         0.88          60


# Accuracy: 0.8833333333333333
