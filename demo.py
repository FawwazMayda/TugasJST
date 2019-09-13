import numpy as np 
from lvq import LVQ
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("iris.csv")
mms = MinMaxScaler()
df['species'] = df['species'].map({'setosa':0,'versicolor':1,'virginica':2})
m = LVQ(n_class=3,input_shape=4,distance='manhattan')
X = mms.fit_transform(df.iloc[:, :4].values)
y = df.iloc[:,-1].values
m.fit(X,y,epoch=1000,lr=0.00005)