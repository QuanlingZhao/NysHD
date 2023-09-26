from mnist import MNIST
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
import copy
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import polynomial_kernel




class hd_adaptive_encoder:
    def __init__ (self,training_dataset,num_lanmark,gamma,dim):
        self.training_dataset = training_dataset
        self.num_landmark = num_lanmark
        self.gamma = gamma
        self.dim = dim

        scale = math.sqrt(1/(2*gamma))
        self.rbf = RBF(scale)

        self.kernel_matrix, self.adaptive_proj = self.generate_adaptive_random_projection()


    def uniform_landmark_selection(self):
        return np.array(random.sample(self.training_dataset.tolist(),self.num_landmark))
    
    def compute_kernel_matrix(self):
        kernel_matrix = polynomial_kernel(self.landmark)

        normed = copy.deepcopy(kernel_matrix)
        
        return normed
    
    def get_unit_random_projection(self):
        #proj = np.random.uniform(low=-1, high=1, size=(self.num_landmark,self.dim))
        #proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        #return proj_norm.T
        #a = np.random.rand(self.dim, self.num_landmark)
        #q, r = np.linalg.qr(a)
        #return q
        proj = np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark))
        proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        return (np.sqrt(0.5 * np.pi)) * proj_norm
        #return np.sign(np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark)))

    def generate_adaptive_random_projection(self):

        self.landmark = self.uniform_landmark_selection()
        kernel_matrix = self.compute_kernel_matrix()
        eigen_values, eigen_vectors = np.linalg.eigh(kernel_matrix)
        eigen_values = eigen_values.real
        eigen_values[eigen_values<=0] = 1e-15
        eigen_vectors = eigen_vectors.real
        random_proj = self.get_unit_random_projection()

        diag = np.diag(1 / np.sqrt(eigen_values), k=0)
        
        eigen = eigen_vectors.T

        adaptive_proj = np.matmul(random_proj,np.matmul(diag,eigen))
        
        return kernel_matrix, adaptive_proj
    

    def adaptive_encode(self,seq):
        features = self.rbf([seq],self.landmark).T
        return (1 / np.sqrt(self.dim)) * np.sign(np.matmul(self.adaptive_proj,features).squeeze())








class hd_model:
    def __init__(self,train_x,train_y,test_x,test_y,num_landmark,gamma,dim,lr):
         self.train_x = train_x
         self.train_y = train_y
         self.test_x = test_x
         self.test_y =test_y
         self.num_landmark = num_landmark
         self.gamma = gamma
         self.dim = dim
         self.lr = lr
         self.encoder = hd_adaptive_encoder(self.train_x,self.num_landmark,self.gamma,self.dim)
         self.class_hvs = np.zeros((10,self.dim))

         self.train_encs = []


    
    def train(self):
        assert len(self.train_x) == len(self.train_y)

        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)

            x = self.train_x[i]
            label = self.train_y[i]    
            enc = self.encoder.adaptive_encode(x)

            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
            softmax = np.exp(similarities) / sum(np.exp(similarities))
            pred = np.argmax(similarities)
            
            self.class_hvs[label] += (1 - similarities[label]) * enc
            self.class_hvs[pred] -= (1 - similarities[pred]) * enc

            self.train_encs.append(enc)
            



    def test(self):
        assert len(self.test_x) == len(self.test_y)
        preds = []
        for i in range(len(self.test_x)):
            if i % 500 == 0:
                print(i)
            x = self.test_x[i]
            label = self.test_y[i]
            enc = self.encoder.adaptive_encode(x)
            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
            pred = np.argmax(similarities)
            preds.append(pred)
        

        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")




    def retrain(self):
        for e in range(10):
            count = 0
            print(e)
            for i in range(len(self.train_encs)):
                enc = self.train_encs[i]
                label = self.train_y[i]
                similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
                pred = np.argmax(similarities)
                if pred != label:
                    self.class_hvs[label] += self.lr * (1 - similarities[label]) * enc
                    self.class_hvs[pred] -= self.lr * (1 - similarities[pred]) * enc
                    count += 1
            
            #train_acc = 1 - (count / 60000)
            #print(train_acc)
            print(count)









if __name__ == "__main__":

    data = MNIST('data')

    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()

    train_x = np.array(train_x) / 255
    train_y = np.array(train_y)
    test_x = np.array(test_x) / 255
    test_y = np.array(test_y)


    #train_x = train_x[0:5000]
    #train_y = train_y[0:5000]
    #test_x = test_x[0:5000]
    #test_y = test_y[0:5000]

    model = hd_model(train_x,train_y,test_x,test_y,3000,0.3,10000,0.5)

    model.train()
    model.test()
    model.retrain()
    model.test()



'''
if __name__ == "__main__":
    data = MNIST('data')
    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()
    train_x = np.array(train_x) / 255
    train_y = np.array(train_y)
    test_x = np.array(test_x) / 255
    test_y = np.array(test_y)

    encoder = hd_adaptive_encoder(train_x,1200,0.1,10000)

    encs = []
    for i in range(25):
        encs.append(encoder.adaptive_encode(train_x[i]))
    
    encs = np.array(encs)

    a = (encs @ encs.T)

    scale = math.sqrt(1/(2*0.1))
    rbf = RBF(scale)
    b = (rbf(train_x[0:25],train_x[0:25]))

    k = rbf(encoder.landmark)
    eigen_values, eigen_vectors = np.linalg.eigh(k)
    eigen_values = eigen_values.real
    eigen_values[eigen_values<=0] = 1e-15
    eigen_vectors = eigen_vectors.real
    diag = np.diag(1 / np.sqrt(eigen_values), k=0)    
    eigen = eigen_vectors.T
    nystrom = rbf(train_x[0:25],encoder.landmark)
    c = nystrom @ eigen.T @ diag @ diag @ eigen @ nystrom.T

    print(sum(sum(a - b)) / (25*25))
    print(sum(sum(c - b)) / (25*25))

    plt.imshow(a)
    plt.colorbar()
    plt.show()

    plt.imshow(b)
    plt.colorbar()
    plt.show()

    plt.imshow(c)
    plt.colorbar()
    plt.show()
'''





































