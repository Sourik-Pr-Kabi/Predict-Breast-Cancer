#Importing the libraries

import numpy as np
import pandas as pd


#Importing the dataset

dataset = pd.read_csv('Breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


#Predicting the Test set results

y_pred = classifier.predict(X_test)

#Un-comment line 32 and line 33 to print the predicted outputs of the test set and the actual results of the test set
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Computing the accuracy with k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


#Input data for Benign or Malignant prediction

a = int(input('Enter Clump Thickness:'))
b = int(input('Enter Uniformity of Cell Size:'))
c = int(input('Enter Uniformity of Cell Shape:'))
d = int(input('Enter Marginal Adhesion:'))
e = int(input('Enter Single Epithelial Cell Size:'))
f = int(input('Enter Bare Nuclei:'))
g = int(input('Enter Bland Chromatin:'))
h = int(input('Enter Normal Nucleoli:'))
i = int(input('Enter Mitoses:'))


#Prediciting the result

out = classifier.predict([[a, b, c, d, e, f, g, h, i]])


#Printing the output

if out == [4]:
    print('Malignant')
elif out == [2]:
    print('Benign')
