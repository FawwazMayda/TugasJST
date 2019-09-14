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
            #print("WL: {} BL: {}".format(wl,bl))
            #print(np.random.rand(layer1,layer2))
            self.weights.append(np.random.rand(layer1,layer2))
            self.dw.append(np.zeros(wl))
            self.bias.append(np.random.rand(layer2,1))
            self.db.append(np.zeros(bl))
        """
        print("self.weight")
        for e in self.weights:
            print("*****")
            print(e)  
        print("self.dw")
        for e in self.dw:
            print("*****")
            print(e)    
        print("self.bias")
        for e in self.bias:
            print("*****")
            print(e)
        print("self.db")
        for e in self.db:
            print("*****")
            print(e)  
        """
        #print("Weights:{} Bias:{} num_layers:{}".format(len(self.weights),len(self.bias),len(self.all_layers)))
    
        

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_backward(self,x):
        output = self.sigmoid(x)
        return output*(1 - output )
    

    def __forward(self,X):
        #print("__forward")
        self.a = [X]
        self.z = [X]

        for (weight,bias) in zip(self.weights,self.bias):
            temp_dot = (np.dot(self.a[-1],weight))
            temp_dot = temp_dot + bias.T

            self.z.append(temp_dot)
            self.a.append(self.sigmoid(temp_dot))
        """
        print("Self.a")
        for i,e in enumerate(self.a):
            
            print("index:{} shape:{}".format(i,e.shape))
        print("Self.z")
        for i,e in enumerate(self.z):
            
            print("index:{} shape:{}".format(i,e.shape))
        """

        return self.a[-1].T


    def backward(self,X,y,learn_rate):
        #print("__backward")
        m = X.shape[0]
        delta_w = []
        delta_b = []
        num_layers = len(self.all_layers)
        #print(y.shape)
        for i in reversed(range(num_layers)):
            if i==num_layers-1:
                #print("This One")
                da = self.a[i] - y
            else:
                #print("This two")
                da = self.weights[i].dot(dz.T).T
                #da = self.weights[i].dot(dz).T
            
            #print("da.shape:{}".format(da.shape))
            dz = da * self.sigmoid_backward(self.z[i])
            #print("dz.shape:{}".format(dz.shape))
            dw = dz.dot(self.a[i].T) 
            dw = dz*self.a[i]*learn_rate
            #print("dw.shape:{}".format(dw.shape))
            db = dz*learn_rate
            db = dz.sum(axis=0)*learn_rate
            #print("db.shape:{}".format(db.shape))

            delta_w.append(dw)
            delta_b.append(db)
        """
        print("INI SHAPE")
        print(len(delta_w))
        print(np.array(delta_w[0]).shape)
        print(np.array(delta_w[1]).shape)
        print(len(delta_b))
        print(np.array(delta_b[0]).shape)
        print(np.array(delta_b[1]).shape)
        print(np.array(delta_b[2]).shape)
        print(np.array(delta_b[3]).shape)
        print(np.array(delta_b[4]).shape)
        
        print("UPDATE")
        
        for (index,dw,db) in zip( reversed(range(num_layers-1)),delta_w,delta_b):
            self.dw[index] = learn_rate*dw
            self.db[index] = learn_rate*db
            self.weights[index] -= self.dw[index]
            self.bias[index] -= self.db[index]
        """
        for i in range(1,num_layers):
            #print(i)
            print("change")
            first = delta_w[i-1]*1
            #print(first)
            second = delta_w[i]*1
            #print(second)
            update_w = np.dot(second.T,first)
            print(update_w)
            self.dw[num_layers-i-1] = update_w
            self.db[num_layers - i-1] = db[i-1]
            self.weights[num_layers - i-1] += self.dw[num_layers-i-1]
            self.bias[num_layers - i-1] += self.db[num_layers-i-1]
            

    def predict(self,X):
            res = self.__forward(X)
            res = np.array(res).T
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
                #print(pred_class)
                acc = ((pred_class == y_test).sum()) / y_test.shape[0]
                print("Epoch: {} Lr: {:.4f} Acc: {:.4f}".format(e+1,learn_rate,acc))
                #print(self.weights[0])



