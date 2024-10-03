# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by:chaithanya.c 
RegisterNumber:2305002004 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')
```

## Output:

![image](https://github.com/user-attachments/assets/ccd294a5-4f8a-4588-a7f5-264540bf9fda)

![image](https://github.com/user-attachments/assets/bce1d281-7cf0-47d3-bc97-a3976129b12d)

![image](https://github.com/user-attachments/assets/8c180a5d-f031-49dc-9c39-ccea17c3574e)

![image](https://github.com/user-attachments/assets/f3de07ae-05ac-43f6-b450-093496cc4707)

![image](https://github.com/user-attachments/assets/7f8ceec2-a3eb-4c97-ab06-7c4fd285ee5a)

![image](https://github.com/user-attachments/assets/81cf5efa-8e71-416a-9c70-dbdcae316677)

![image](https://github.com/user-attachments/assets/b3268572-c12a-4952-82c6-771f95423e1a)

![image](https://github.com/user-attachments/assets/dad0382d-199e-4453-9146-1f10bdbbe293)

![image](https://github.com/user-attachments/assets/942537af-9ca5-4265-af9d-3b7477ed4ce6)












## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
