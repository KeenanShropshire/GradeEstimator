import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep = ";")
print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", ]]
print(data.head())

predict = "G3" 

x = np.array(data.drop([predict], 1)) 
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train) 
acc = linear.score(x_test, y_test) #returns a value that represents the accuracy of the model
print(acc)

print("Coef: \n" , linear.coef_)
print("Intercept: \n" , linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
