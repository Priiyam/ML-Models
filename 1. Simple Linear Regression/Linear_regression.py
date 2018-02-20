# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression #importing class
regressor = LinearRegression() #creating an object of class, regressor
regressor.fit(X_train, y_train) #fit our regressor object to training data

# Predicting the test set results
y_pred = regressor.predict(X_test) #y_pred -> vector of prediction of dependent variable. Will contain predicted salaries for all test set

# Visualising Training set Result
plt.scatter(X_train, y_train, color='red') #training set line
plt.plot(X_train, regressor.predict(X_train), color='blue') #Simple Linear Regression fitting line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising Test Result
plt.scatter(X_test, y_test, color='red') #training set line
plt.plot(X_train, regressor.predict(X_train), color='blue') #Simple Linear Regression fitting line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()