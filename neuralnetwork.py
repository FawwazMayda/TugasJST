import numpy as np 

class NeuralNetwork():

    def __init__(self,n_class,input_shape,hidden_layers):
        self.n_class = n_class
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.all_layers = []
        self.all_layers.append(input_shape)
        for e in hidden_layers:
            self.all_layers.append(e)
        self.all_layers.append(self.n_class)
        #self.all_layers = self.hidden_layers.copy.append(self.n_class).append(self.input_shape,0)
        self.weights = []
        self.bias = []
        self.dw = [] #Derivate for weights
        self.db = [] #Derivati for Bias
        self.a = [] #layers output after activation
        self.z = [] #layers output before activation
        print(self.all_layers)
        for layer1,layer2 in zip(self.all_layers[:-1], self.all_layers[1:]):
            wl = (layer1,layer2)
            bl = (layer2,1)
            print("WL: {} BL: {}".format(wl,bl))
            print(np.random.rand(layer1,layer2))
            self.weights.append(np.random.rand(layer1,layer2))
            self.dw.append(np.zeros(bl))
            self.bias.append(np.random.rand(layer1,layer2))
            self.db.append(np.zeros(bl))

    
        

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    def sigmoid_backward():
        return self.sigmoid(x)*(1- self.sigmoid(x))
    

    def __forward(self,X):
        self.a = [X.T]
        self.z = [None]

        for (weight,bias) in zip(self.weights,self.bias):
            self.z.append(np.dot(self.a[-1],weight) + bias)
            self.a.append(sigmoid(self.z[-1]))

        return self.a[-1].T


    def backward(self,X,y,learn_rate):
        m = X.shape[0]
        delta_w = []
        delta_b = []
        num_layers = len(self.all_layers)
        for i in reversed(range(num_layers)):
            if i==num_layers:
                da = self.a[i] - y.T
            else:
                da = self.weights[i].T.dot(dz)
            dz = da * sigmoid_backward(self.z[i])
            dw = dz.dot(self.a[i - 1].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            delta_w.append(dw)
            delta_b.append(db)

        for (index,dw,db) in zip( reversed(range(num_layers)),delta_w,delta_b):
            self.dw[index] = learn_rate*dw
            self.db[index] = learn_rate*db
            self.weights[index] -= self.dw[index]
            self.bias[index] -= self.db[index]

    def predict(self,X):
            res = self.__forward(X)
            return [ self.__inverse_encode(e) for e in res]

    def predict_proba(self,X):
            return self.__forward(X)

    def __encode(self,y):
            y_encode = []
            for e in y:
                k = np.zeros(self.n_class)
                k[e] = 1
                y_encode.append(k)
            return np.array(y_encode)

    def __inverse_encode(self,y):
            return np.argmax(y)

    def fit(self,X,y,epochs,learn_rate,eval_set):
            y = self.__encode(y)
            for e in range(epochs):
                self.__forward(X)
                self.backward(X,y,learn_rate)
                X_test,y_test = eval_set
                pred_class = self.predict(X_test)
                acc = ((pred_class == y_test).sum()) / y_test.shape[0]
                print("Epoch: {} Lr: {:.4f} Acc: {:.4f}".format(e+1,lr,acc))



