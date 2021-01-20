import pandas as pd

# Get dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:, -1].values

# Label Encoding categorial data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One hot encoding 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)