import numpy as np 

class LVQ():

    def distance(self,vector_1,vector_2):
        if self.dist=='euclidean':
            return np.sqrt(np.sum(np.square(vector_1 - vector_2)))  
        elif self.dist=='manhattan':
            return np.sum(np.abs(vector_1 - vector_2))      

    def __best_ref_vector(self,x):
        rs = [ self.distance(x,self.ref_vector[i]) for i in range(self.n_class)]
        rs = np.array(rs)
        return np.argmin(rs)
    
    def __init__(self,n_class,input_shape,distance='euclidean'):
        self.n_class = n_class
        self.input_dim = input_shape
        self.ref_vector = np.zeros((n_class, input_shape))
        self.dist=distance
        

    def __train(self,X,y,lr):
        for i in range(len(y)):
            C = y [i]
            idx = self.__best_ref_vector(X[i])
            if idx == C:
                self.ref_vector[idx] = self.ref_vector[idx] + lr*(X[i] - self.ref_vector[idx])
            else:
                self.ref_vector[idx] = self.ref_vector[idx] - lr*(X[i] - self.ref_vector[idx])


    def fit(self,X,y,epoch,lr,eval_set):
        # X, y is Np Array
        for i in range(self.n_class):
            idx = np.where(y==i)[0][0]
            self.ref_vector[i] = X[idx]
            X = np.delete(X,idx,0)
            y = np.delete(y,idx,0)

        for e in range(epoch):
            self.__train(X,y,lr)
            y_pr = self.predict(eval_set[0])
            acc = ((y_pr == eval_set[1]).sum()) / y.shape[0]
            print("Epoch: {} Lr: {:.4f} Acc: {:.4f}".format(e+1,lr,acc))
            #lr = lr *(1.0-(e/epoch))
            #print(self.ref_vector)

    def predict(self,X_new):
        return np.array([ self.__best_ref_vector(X_new[i]) for i in range(X_new.shape[0])])


        
        
        

    