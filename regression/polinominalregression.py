import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
## Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualising the Linear Regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Polinominal Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results (for higher resolution and smoother curve)
 
# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))