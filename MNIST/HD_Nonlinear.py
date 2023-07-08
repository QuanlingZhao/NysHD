from mnist import MNIST
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import math





class linear_random_projection:
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_projection()

    def init_projection(self):
        print("enter")
        #self.projection = np.random.multivariate_normal(mean=np.zeros(self.out_dim), cov=2* (1/self.in_dim) *np.eye(self.out_dim), size=self.in_dim)
        self.projection = np.random.normal(0, 1, size=(self.out_dim, self.in_dim)).T
        print("exit")
        self.bias = np.random.uniform(low=0, high=2*np.pi, size=(1,self.out_dim))

    def encode(self,x):
        return (np.cos(x @ self.projection + self.bias) / np.sqrt(self.out_dim)).squeeze()


class hd_model:
    def __init__(self,train_x,train_y,test_x,test_y,in_dim,out_dim):
         self.train_x = train_x
         self.train_y = train_y
         self.test_x = test_x
         self.test_y =test_y
         self.in_dim = in_dim
         self.out_dim = out_dim

         self.encoder = linear_random_projection(self.in_dim,self.out_dim)

         self.class_hvs = np.zeros((10,self.out_dim))

         self.train_enc = []
         self.test_enc = []
    
    def train(self):
        assert len(self.train_x) == len(self.train_y)
        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)
            x = self.train_x[i]
            label = self.train_y[i]    
            enc = self.encoder.encode(x)
            self.train_enc.append(enc)
            self.class_hvs[label] += enc
    
    def test(self):
        assert len(self.test_x) == len(self.test_y)
        preds = []
        for i in range(len(self.test_x)):
            if i % 500 == 0:
                print(i)
            x = self.test_x[i]
            label = self.test_y[i]
            enc = self.encoder.encode(x)
            self.test_enc.append(enc)
            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)
            pred = np.argmax(similarities)
            preds.append(pred)
        
        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")


    def retrain(self):
        print("====Retrain")
        for i in range(len(self.train_enc)):
            similarities = cosine_similarity(self.train_enc[i].reshape(1, -1), self.class_hvs)
            pred = np.argmax(similarities)
            if pred != self.train_y[i]:
                self.class_hvs[pred] -= self.train_enc[i]
                self.class_hvs[self.train_y[i]] += self.train_enc[i]

        preds = []
        for i in range(len(self.test_enc)):
            similarities = cosine_similarity(self.test_enc[i].reshape(1, -1), self.class_hvs)
            preds.append(np.argmax(similarities))
        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")






if __name__ == "__main__":

    data = MNIST('data')

    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()

    train_x = np.array(train_x) / 255
    train_y = np.array(train_y)
    test_x = np.array(test_x) / 255
    test_y = np.array(test_y)

    #train_x = (train_x - 0.1307) / 0.3081
    #test_x = (test_x - 0.1307) / 0.3081



    model = hd_model(train_x,train_y,test_x,test_y,28*28,10000)

    model.train()
    model.test()




    for i in range(10):
        print(i)
        model.retrain()


