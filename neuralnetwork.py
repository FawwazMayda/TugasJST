import numpy as np 

class NeuralNetwork():

    def __init__(self,n_class,input_shape,hidden_layers):
        self.n_class = n_class
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.all_layers = self.hidden_layers.copy().append(n_class).append(self.input_shape,0)
        self.weights = []
        self.bias = []
        self.dw = []
        self.db = []
        self.a = [] #layers output after activation
        self.z = [] #layers output before activation

        for layer1,layer2 in zip(self.all_layers[:-1], self.all_layers[1:]):
            wl = (layer1,layer2)
            bl = (layer2,1)
            self.weights.append(np.random.normal(wl))
            self.dw.append(np.zeros(wl))
            self.bias.append(np.random.normal(bl))
            self.db.append(np.zeros(bl))
    

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    

    def forward(self,X):
        self.a = [X.T]
        self.z = [None]

        for (weight,bias) in zip(self.weights,self.bias):
            self.z.append(np.dot(self.a[-1],weight) + bias)
            self.a.append(sigmoid(self.z.[-1]))
        return self.a[-1].T


    def backward(self):

