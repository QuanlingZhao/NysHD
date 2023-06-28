from gappy_kernel import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, precision_score
from Bio import SeqIO
from scipy.sparse import coo_matrix, vstack
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics.pairwise import cosine_similarity
from get_data import *



class hd_adaptive_encoder:
    def __init__ (self,training_dataset,num_lanmark,k,g,t,dim):
        self.training_dataset = training_dataset
        self.num_landmark = num_lanmark
        self.k = k
        self.g = g
        self.t = t
        self.dim = dim

        self.kernel_matrix, self.adaptive_proj, self.landmark_features = self.generate_adaptive_random_projection()

    def uniform_landmark_selection(self):
        self.landmark = random.sample(self.training_dataset,self.num_landmark)
    
    def compute_kernel_matrix(self):
        landmark_featuren = gappypair_kernel(self.landmark, k=self.k, g=self.g, t=self.t, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        return np.matmul(landmark_featuren, np.transpose(landmark_featuren)), landmark_featuren
    
    def get_unit_random_projection(self):
        proj = np.random.uniform(low=-1, high=1, size=(self.num_landmark,self.dim))
        proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        return proj_norm.T

    def generate_adaptive_random_projection(self):
        landmarks = self.uniform_landmark_selection()
        kernel_matrix, landmark_features = self.compute_kernel_matrix()
        eigen_values, eigen_vectors = np.linalg.eig(kernel_matrix)
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real
        
        random_proj = self.get_unit_random_projection()
        diag = np.diag(1 / np.sqrt(eigen_values), k=0)
        
        assert np.sum(np.isnan(diag)) == 0
        
        eigen = eigen_vectors.T
        adaptive_proj = np.matmul(random_proj,np.matmul(diag,eigen))
        
        return kernel_matrix, adaptive_proj, landmark_features
    

    def adaptive_encode(self,seq):
        features = gappypair_kernel([seq], k=self.k, g=self.g, t=self.t, include_flanking=True, gapDifferent = False, sparse = True).toarray().T
        return np.squeeze(np.matmul(self.adaptive_proj,np.matmul(self.landmark_features,features)))





class hd_adaptive_model:
    def __init__(self, dim, num_lanmark, k, g, t, train_x, train_y, test_x, test_y):
        self.dim = dim

        self.encoder = hd_adaptive_encoder(train_x,num_lanmark,k,g,t,dim)
        self.class_hvs = np.zeros((6,self.dim))
        
        self.train_x = train_x
        self.train_y = train_y
        self.train_encs = []

        self.test_x = test_x
        self.test_y = test_y
        self.test_encs = []

    def train(self):
        assert len(self.train_x) == len(self.train_y)
        
        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)
            
            sequence = self.train_x[i].upper()
            label = self.train_y[i]    
            
            enc = self.encoder.adaptive_encode(sequence)
            self.train_encs.append(enc)
            self.class_hvs[label] += enc

        
    def test(self):
        assert len(self.test_x) == len(self.test_y)
        
        preds = []
        
        for i in range(len(self.test_x)):
            if i % 500 == 0:
                print(i)
            
            sequence = self.test_x[i].upper()
            label = self.test_y[i]
                        

            enc = self.encoder.adaptive_encode(sequence)
            self.test_encs.append(enc)
                
            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)
            pred = np.argmax(similarities)
            preds.append(pred)
        
        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")


if __name__ == "__main__":
    random.seed(3)
    np.random.seed(3)

    X_train, X_test, y_train, y_test = get_data(0.1,42)

    X_train, X_test, y_train, y_test = get_data(0.1,42)

    model = hd_adaptive_model(10000, 50, 3, 0, 2, X_train, y_train, X_test, y_test)

    print(model.encoder.kernel_matrix)
    print(model.encoder.adaptive_proj)
    print(model.encoder.landmark_features)
    print(model.encoder.kernel_matrix.shape)
    print(model.encoder.adaptive_proj.shape)
    print(model.encoder.landmark_features.shape)



    model.train()

    model.test()










































