<b>Predict Breast Cancer</b>

This project uses Logistic Regression to predict based on some inputs whether the cancer of the person under consideration is Benign or Malignant.

The repository consists of:
  1. A .ipynb file
  2. A .py file
  3. A .csv file

The .py file is simply the python implementation of the Jupyter Notebook file.
The .csv file contains the dataset of many patients, each record consisting of the following features:
  1. Clump Thickness
  2. Uniformity of Cell Size
  3. Uniformity of Cell Shape
  4. Marginal Adhesion
  5. Single Epithelial Cell Size
  6. Bare Nuclei
  7. Bland Chromatin
  8. Normal Nucleoli
  9. Mitoses
 10. Class

There is no missing data in the dataset. 80% of the data is used to train the Logistic Regression model and the remaining is used in testing.
K-fold cross validation is used to measure the accuracy and the outcome is - the model is 96.7 % accurate with a standard deviation of 1.97 %.
