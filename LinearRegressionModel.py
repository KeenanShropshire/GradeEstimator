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

predict = "G3" #predict is G3 because you want to predict the final grade; also known as a label

x = np.array(data.drop([predict], 1)) #return a new data frame that doesnt include the predict value
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train) #find best fit line using x_train and y_train data and store that line in linear
acc = linear.score(x_test, y_test) #returns a value that represents the accuracy of the model
print(acc)

print("Coef: \n" , linear.coef_)
print("Intercept: \n" , linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])