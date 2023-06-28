import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from get_data import *
from sklearn.metrics.pairwise import cosine_similarity



class hd_n_gram_encoder:
    def __init__(self,dim,n):
        self.dim = dim
        self.n = n
        self.init_codebook()

    def init_codebook(self):
        alphabet = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            alphabet[letter] = np.random.randint(2, size=self.dim)
        self.alphabet = alphabet
    
    def n_gram_encode(self,sequence):
        sequence_length = len(sequence)
        enc = np.zeros(self.dim)
        for i in range(sequence_length - self.n + 1):
            n_gram = sequence[i:i+self.n]
            gram_enc = np.zeros(self.dim)
            for letter in range(self.n):
                gram_enc += np.roll(self.alphabet[n_gram[letter]],letter)
            enc += (gram_enc).astype('i')
        return (enc).astype('i')




class hd_n_gram_model:
    def __init__(self, dim, n, train_x, train_y, test_x, test_y):
        self.dim = dim
        self.n = n
        self.encoder = hd_n_gram_encoder(self.dim, self.n)
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
            
            enc = self.encoder.n_gram_encode(sequence)
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
                        

            enc = self.encoder.n_gram_encode(sequence)
            self.test_encs.append(enc)
            
            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)
            pred = np.argmax(similarities)
            preds.append(pred)
        
        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(0.1,42)

    model = hd_n_gram_model(10000, 3, X_train, y_train, X_test, y_test)

    model.train()

    model.test()









































