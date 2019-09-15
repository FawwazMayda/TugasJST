import numpy as np

class Layer():
    def __init__(self,input_shape,n_neuron,activation='linear'):
        self.input_shape=input_shape
        self.n_neuron = n_neuron
        self.weights = np.random.rand(self.input_shape,self.n_neuron)
        self.bias = np.random.rand(self.n_neuron)
        self.delta= 0
        self.activation = activation
        self.after_activation = None
        self.layer = None

    def apply_activation(self,x,grad=False):

        if grad:
            if self.activation=='sigmoid':
                output = self.apply_activation(x)
                return output*(1.0-output)
            elif self.activation=='tanh':
                output = self.apply_activation(x)
                return 1 - output**2
            elif self.activation=='relu':
                return self.apply_activation(x)
            else:
                return 1
            

        if self.activation=='sigmoid':
            return 1/(1+np.exp(-x))
        if self.activation=='tanh':
            return np.tanh(x)
        if self.activation=='relu':
            return np.maximum(0,x)

        return x
        
    def activate(self,x):
        r = np.dot(x,self.weights) + self.bias
        self.after_activation = self.apply_activation(r)
        return self.after_activation

