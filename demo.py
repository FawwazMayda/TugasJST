import numpy as np 
from lvq import LVQ
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from neuralnet import Neural
from layer import Layer

df = pd.read_csv("iris.csv")
mms = MinMaxScaler()
df['species'] = df['species'].map({'setosa':0,'versicolor':1,'virginica':2})

m = LVQ(n_class=3,input_shape=4,distance='euclidean')
X = mms.fit_transform(df.iloc[:, :4].values)
y = df.iloc[:,-1].values
#TTS
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
nn = Neural(3)
nn.add_layer(Layer(4,5,'relu'))
nn.add_layer(Layer(5,3,'sigmoid'))
angka = [[4,5,6,7],[10,20,30,40],[12,76,34,23],[5,5,5,8]]
angka = np.array(angka)
print(angka.shape)
print(nn.predict(angka))
nn.fit(X_train,y_train,epochs=1000,learn_rate=0.01)
#nn.fit(X_train,y_train,epochs=2,learn_rate=100,eval_set=(X_test,y_test))
