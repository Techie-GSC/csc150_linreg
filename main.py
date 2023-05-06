# Adapated from https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt  # Tools to create a graphical plot of the data
import pandas as pd  # Tools to read in the data from a CSV file
from sklearn.linear_model import LinearRegression  # The particular mathematical model to use

data = pd.read_csv('student_scores.csv')  # load data from the designated file
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into an array
Y = data.iloc[:, 1].values.reshape(
    -1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression(
)  # create object for the mathematical model
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(
    X)  # make predictions based on the data in the file

# Create a scatterplot showing the data and the calculated regression line (the "predictions")
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('Line indicates model predictions')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.savefig('graph.png')  # have to save as an image when using repl.it

# Predict a student's score based on an input study time
hoursStudied = float(
    input("How many hours did you spend studying for the test? "))

# This line puts the value of hoursStudied into the linear regression model and predicts the test score
prediction = linear_regressor.predict([[hoursStudied]])

print("Because you studied for", hoursStudied,
      "hours, the linear regression model predicts your score will be",
      round(float(prediction), 2))
