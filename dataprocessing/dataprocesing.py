#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import data set
dataset = pd.read_csv("./Data.csv")
    #feature extraction
x = dataset.iloc[:, :-1].values
    #vector of depentend variable
y = dataset.iloc[:,-1].values

#take care of missing data
from sklearn.impute import SimpleImputer
 #put average value of other in mising value
imputer = SimpleImputer(missing_values=np.nan , strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#Encoding categorical data

    #Encoding the Independent Variable
    #encode for that Column that havent numerical value then transform to the form of numerical value that havent raltion to eacother
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

    #Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y = le.fit_transform(y)

#Splitting the dataset into the Training set and Test set



#Feature Scaling

