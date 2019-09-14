import numpy as np

class Layer():
    def __init__(self,input_shape,n_neuron,activation):
        self.input_shape=input_shape
        self.n_neuron = n_neuron
        self.weights = np.random.rand(self.input_shape,self.n_neuron)
        self.bias = np.random.rand(self.n_neuron)
        self.delta= 0
        self.activation = activation

    def __apply_activation(self,x,grad=False):

        if grad:
            

        if self.activation=='sigmoid':
            return 1/(1+np.exp(-x))
        if self.activation=='tanh':
            return np.tanh(r)
        if self.activation=='relu':
            return np.max(0,x)

        return x
        
    def activation(self,x):
        r= np.dot(self.weights,x) + bias
        return self.__apply_activation(r)

