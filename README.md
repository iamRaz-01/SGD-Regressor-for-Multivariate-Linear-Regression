# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data. 
2. Print the placement data and salary data. 
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices. 
5. Display the results

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Abdul Rasak N 
RegisterNumber:  24002896
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = fetch_california_housing()
X = data.data[:, :3]  # Select the first three features for X
y = np.column_stack((data.target, data.data[:, 6]))  # Stack the target and seventh feature for y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scalers for X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scale the training and testing data
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Initialize the model and train it
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, y_train)

# Predict and inverse-transform the predictions and true values
y_pred = multi_output_sgd.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Display the results
print(y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", y_pred[:5])

```

## Output:
![image](https://github.com/user-attachments/assets/ce7def94-3dc9-47e4-822d-a8acb9eb3fd6)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
