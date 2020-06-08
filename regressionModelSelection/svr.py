import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sclaer_x = StandardScaler()
sclaer_y = StandardScaler()
x_train = sclaer_x.fit_transform(x_train)
y_train = sclaer_y.fit_transform(y_train)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

y_pred = sclaer_y.inverse_transform(regressor.predict(sclaer_x.transform(x_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
