# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file.

2.Load and Prepare Data.Load the dataset.Extract and scale features and target values.

3.Train Linear RegressionInitialize parameters.Perform gradient descent to update parameters.

4.Make PredictionScale new input data.

5.Predict using trained model.Inverse scale the prediction.

6.Print the predicted value.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Lakshmi Priya.V
RegisterNumber: 212223220049 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions -y).reshape(-1,1)
        theta -= learning_rate *(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/58e81f5f-6ff6-4218-979d-62de612278a4)

![image](https://github.com/user-attachments/assets/f1ea942f-304d-4283-903e-14248d25e3c0)

![image](https://github.com/user-attachments/assets/b5672110-eedf-425a-b852-23a34613449c)

![image](https://github.com/user-attachments/assets/48462c1d-1851-4709-9cff-45110bccb866)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
