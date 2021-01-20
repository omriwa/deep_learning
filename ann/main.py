import pandas as pd
import tensorflow as tf

# Get dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values