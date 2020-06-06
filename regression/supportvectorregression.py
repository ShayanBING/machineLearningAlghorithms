import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sclaer_x = StandardScaler()
sclaer_y = StandardScaler()
x = sclaer_x.fit_transform(x)
y = sclaer_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

print(sclaer_y.inverse_transform(regressor.predict(sclaer_x.transform([[6.5]]))))

plt.scatter(sclaer_x.inverse_transform(x),sclaer_y.inverse_transform(y),color='red')
plt.plot(sclaer_x.inverse_transform(x),sclaer_y.inverse_transform(regressor.predict(x)),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()

#smoother plot
X_grid = np.arange(min(sclaer_x.inverse_transform(x)), max(sclaer_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sclaer_x.inverse_transform(x), sclaer_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sclaer_y.inverse_transform(regressor.predict(sclaer_x.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()









