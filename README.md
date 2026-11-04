# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.

## Program:
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
df = pd.read_csv(r"C:\Users\admin\Downloads\exp_2_dataset_student_scores.csv")
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
print("Developed by Gayathri D")

X = df.iloc[:, :-1].values   
Y = df.iloc[:, -1].values    
print("X (features):", X.flatten())
print("Y (targets):", Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

new_hours = np.array([[2.5], [8.0]])   
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```

## Output:
<img width="940" height="96" alt="Screenshot 2026-11-04 134344" src="https://github.com/user-attachments/assets/1ee48dac-edf9-4b6c-83a4-5cfddac0757d" />

<img width="882" height="57" alt="Screenshot 2026-11-04 134355" src="https://github.com/user-attachments/assets/c1c99f93-422b-4e34-a49c-064cdcbb1eb9" />

<img width="662" height="491" alt="Screenshot 2026-11-04 134412" src="https://github.com/user-attachments/assets/c50ea398-8bc2-4eb4-a3d6-562fb3434a26" />

<img width="660" height="488" alt="Screenshot 2026-11-04 134423" src="https://github.com/user-attachments/assets/9cb0c120-7690-4ee3-bb3a-ee1ae866b08e" />

<img width="718" height="82" alt="Screenshot 2026-11-04 134442" src="https://github.com/user-attachments/assets/b07aedd2-4567-4f7d-bbbd-99c7c7399a64" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
