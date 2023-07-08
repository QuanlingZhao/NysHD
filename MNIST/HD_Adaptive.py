from mnist import MNIST
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
import copy
from sklearn.gaussian_process.kernels import RBF





class hd_adaptive_encoder:
    def __init__ (self,training_dataset,num_lanmark,gamma,dim):
        self.training_dataset = training_dataset
        self.num_landmark = num_lanmark
        self.gamma = gamma
        self.dim = dim

        scale = math.sqrt(1/(2*gamma))
        self.rbf = RBF(scale)

        self.kernel_matrix, self.adaptive_proj = self.generate_adaptive_random_projection()

        #print(self.kernel_matrix)
        #print(self.adaptive_proj)
        #print(self.landmark)
        #print("===========================")
        #print(self.kernel_matrix.shape)
        #print(self.adaptive_proj.shape)
        #print(self.landmark.shape)


    def uniform_landmark_selection(self):
        return np.array(random.sample(self.training_dataset.tolist(),self.num_landmark))
    
    def compute_kernel_matrix(self):
        kernel_matrix = self.rbf(self.landmark)
        normed = copy.deepcopy(kernel_matrix)
        
        #for i in range(self.num_landmark):
        #    for j in range(self.num_landmark):
        #        normed[i][j] = normed[i][j] / (np.sqrt(kernel_matrix[i][i]) * np.sqrt(np.sqrt(kernel_matrix[j][j])))

        return normed
    
    def get_unit_random_projection(self):
        proj = np.random.uniform(low=-1, high=1, size=(self.num_landmark,self.dim))
        proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        return proj_norm.T

    def generate_adaptive_random_projection(self):

        while True:
            self.landmark = self.uniform_landmark_selection()
            kernel_matrix = self.compute_kernel_matrix()
            eigen_values, eigen_vectors = np.linalg.eig(kernel_matrix)
            eigen_values = eigen_values.real
            eigen_vectors = eigen_vectors.real
        
            random_proj = self.get_unit_random_projection()
            diag = np.diag(1 / np.sqrt(eigen_values), k=0)
        
            if np.sum(np.isnan(diag)) != 0:
                print("Resampling....")
                continue

            break
        
        eigen = eigen_vectors.T
        adaptive_proj = np.matmul(random_proj,np.matmul(diag,eigen))
        
        return kernel_matrix, adaptive_proj
    

    def adaptive_encode(self,seq):
        features = self.rbf([seq],self.landmark).T
        return np.matmul(self.adaptive_proj,features).squeeze()









class hd_model:
    def __init__(self,train_x,train_y,test_x,test_y,num_landmark,gamma,dim):
         self.train_x = train_x
         self.train_y = train_y
         self.test_x = test_x
         self.test_y =test_y
         self.num_landmark = num_landmark
         self.gamma = gamma
         self.dim = dim

         self.encoder = hd_adaptive_encoder(self.train_x,self.num_landmark,self.gamma,self.dim)

         self.class_hvs = np.zeros((10,self.dim))

         self.train_enc = []
         self.test_enc = []
    
    def train(self):
        assert len(self.train_x) == len(self.train_y)
        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)
            x = self.train_x[i]
            label = self.train_y[i]    
            enc = self.encoder.adaptive_encode(x)
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
            enc = self.encoder.adaptive_encode(x)
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



    model = hd_model(train_x,train_y,test_x,test_y,2000,0.01,10000)

    model.train()
    model.test()




    for i in range(10):
        print(i)
        model.retrain()





































