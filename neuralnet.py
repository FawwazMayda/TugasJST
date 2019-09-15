import numpy as np
from layer import Layer

class Neural():
    def __init__(self,n_class):
        self.layers= []
        self.n_class = n_class

    def y_encode(self,y):
        return np.argmax(y,axis=1)

    def __encode(self,y):
        y_encode = []
        for e in y:
            k = np.zeros(self.n_class)
            k[e] = 1
            y_encode.append(k)
        return np.array(y_encode)

    def add_layer(self,layer):
        self.layers.append(layer)

    def __forward(self,X):
        for layer in self.layers:

            X= layer.activate(X)
        return X

    def predict(self,X):
        X= self.__forward(X)
        return np.argmax(X,axis=1)
    
    def __backward(self,X,y,learn_rate):

        output = self.__forward(X)

        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]

            if current_layer ==self.layers[-1]:
                current_layer.error = y - output
                current_layer.delta = current_layer.error * current_layer.apply_activation(output,grad=True)
            else:
                next_layer = self.layers[i+1]
                current_layer.error = np.dot(next_layer.weights,next_layer.delta)
                current_layer.delta = current_layer.error * current_layer.apply_activation(current_layer.after_activation)
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self.layers[i - 1].after_activation)
            layer.weights += layer.delta * input_to_use.T * learn_rate
           

    def fit(self,X,y,epochs,learn_rate):
        y_encode=self.__encode(y)
        mses = 0
        for e in range(epochs):
            for i in range(len(X)):
                self.__backward(X[i],y_encode[i],learn_rate)
            #print(self.layers[0].weights)
            y_pr = self.__forward(X)
            mse = np.mean(np.square(y_encode-y_pr))
            y_pr_class = self.y_encode(y_pr)
            acc = (y==y_pr_class).sum() /len(y)
            print("MSE:{} ACC:{}".format(mse,acc))
            #print("Epoch:{} Acc:{}".format(e,acc))
