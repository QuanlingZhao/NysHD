from mnist import MNIST
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import math
import random
import pickle
from grakel.datasets import fetch_dataset
from sklearn.model_selection import train_test_split
from grakel.kernels import PropagationAttr
from grakel.kernels import Propagation
import scipy
from Bio.Seq import Seq
from scipy.sparse import csr_matrix
import re

############################################################################
############################################################################
############################################################################
sequenceTypes={'dna':0,'rna':1,'protein':2,'text':3}
alphabets=['ACGT','ACGU','ABCDEFGHIJKLMNOPQRSTUVWXYZ','ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']

def get_numbers_for_sequence(sequence,t=0,reverse=False):
    try:
        ori=[alphabets[t].index(x) for x in sequence]
    except ValueError:
        return [-1]
    if reverse:
        rev=[alphabets[t].index(x) for x in sequence.reverse_complement()]
        if ori>rev:
            return rev
    return ori


def _extract_gappy_sequence(sequence, k, g,t,reverse=False):
    """Compute gappypair-spectrum for a given sequence, k-mer length k and
    gap length g. A 2*k-mer with gap is saved at the same position as a 2*k-mer
    without a gap.
    The idea is to first create a vector of the given size (4**(2*k)) and then
    transform each k-mer to a sequence of length k of numbers 0-3
    (0 = A, 1 = C, 2 = G, 3 = U).
    From there, we can multiply that sequence with a vector of length 2*k,
    containing the exponents of 4 to calculate the position in the spectrum.
    Example: AUUC -> 0331 -> 4**0*1 + 4**1*3 + 4**2*3 + 4**3*0
    """
    n = len(sequence)
    kk=2*k
    alphabet=len(alphabets[t])
    powersize=np.power(alphabet, (kk))
    multiplier = np.power(alphabet, range(kk))[::-1]
    if reverse:
        powersize=int(np.power(alphabet, (kk))/2)

    spectrum = np.zeros((powersize))
    for pos in range(n - kk + 1):
            pos_in_spectrum = np.sum(multiplier * get_numbers_for_sequence(sequence[pos:pos+(kk)],t,reverse=reverse))
            spectrum[pos_in_spectrum] += 1
            for gap in range(1,g+1):
                if (pos+gap+kk)<=n:
                    pos_gap = np.sum(multiplier * get_numbers_for_sequence(sequence[pos:pos+k] + sequence[pos+k+gap:pos+gap+kk],t,reverse=reverse))
                    spectrum[pos_gap] += 1
    return spectrum

def _extract_spectrum_sequence(sequence, k,t,reverse=False):
    """Compute k-spectrum for a given sequence, k-mer length k.
    This method computes the spectrum for a given sequence and k-mer-length k.
    The idea is to first create a vector of the given size (4**k) and then
    transform each k-mer to a sequence of length k of numbers 0-3
    (0 = A, 1 = C, 2 = G, 3 = U).
    From there, we can multiply that sequence with a vector of length k,
    containing the exponents of 4 to calculate the position in the spectrum.
    Example: AUUC -> 0331 -> 4**0*1 + 4**1*3 + 4**2*3 + 4**3*0
    """
    n = len(sequence)
    alphabet=len(alphabets[t])
    spectrum = np.zeros(np.power(alphabet, k))
    multiplier = np.power(alphabet, range(k))[::-1]
    for pos in range(n - k + 1):
            pos_in_spectrum = np.sum(multiplier * get_numbers_for_sequence(sequence[pos:pos+k],t,reverse))
            spectrum[pos_in_spectrum] += 1
    return spectrum

def _extract_gappy_sequence_different(sequence, k, g,t,reverse=False):
    """Compute gappypair-spectrum for a given sequence, k-mer length k and
    gap length g. A 2*k-mer with gap is saved at the same position as a 2*k-mer
    without a gap. A 2*k-mer with a certain gap size is saved at a different
    position than the same 2*k-mer with no gaps or another number of gaps.
    """
    n = len(sequence)
    kk=2*k
    alphabet=len(alphabets[t])
    powersize=np.power(alphabet, (kk))
    multiplier = np.power(alphabet, range(kk))[::-1]
    if reverse:
        powersize=int(np.power(alphabet, (kk))/2)

    spectrum = np.zeros((g+1)*(powersize))
    for pos in range(n - kk + 1):
            pos_in_spectrum = np.sum(multiplier * get_numbers_for_sequence(sequence[pos:pos+(kk)],t,reverse=reverse))
            spectrum[pos_in_spectrum] += 1
            if (pos+g+kk+1)<n:
                for gap in range(1,g+1):
                    pos_gap = np.sum(multiplier * get_numbers_for_sequence(sequence[pos:pos+k] + sequence[pos+k+gap:pos+gap+kk],t,reverse=reverse))
                    spectrum[(gap*(powersize))+pos_gap] += 1
    return spectrum

def gappypair_kernel(sequences, k, g,t,sparse=True, reverse=False, include_flanking=False, gapDifferent = True):
    """Compute gappypair-kernel for a set of sequences using k-mer length k
    and gap size g. The result than can be used in a linear SVM or other
    classification algorithms.
    Parameters:
    ----------
    sequences:              A list of Biopython sequences
    k:                      Integer. The length of kmers to consider
    g:                      Integer. Gapps allowed. 0 by default.
    t:                      Which alphabet according to sequenceTypes.
                            Assumes Dna (t=0).
    sparse:                 Boolean. Output as sparse matrix? True by default.
    reverse:                Boolean. Reverse complement taken into account?
                            False by default.
    include_flanking:       Boolean. Include flanking regions?
                            (the lower-case letters in the sequences given)
    gapDifferent:           Boolean. If k-mers with different gaps should be
                            threated differently or all the same.
                            True by default.
    Returns:
    -------
    A numpy array of shape (N, 4**k), containing the k-spectrum for each
    sequence. N is the number of sequences and k the length of k-mers considered.
    """

    spectrum = []
    for seq in sequences:
    # To be capable to handle string input - does that make sense?
    #seq=Seq(seq)
        if include_flanking:
            seq = seq.upper()
        else:
            seq = [x for x in seq if 'A' <= x <= 'Z']
        if (g>0) and gapDifferent:
            spectrum.append(_extract_gappy_sequence_different(seq, k, g, t))
        elif g>0:
            spectrum.append(_extract_gappy_sequence(seq, k, g, t))
        else:
            spectrum.append(_extract_spectrum_sequence(seq, k, t))
    if sparse:
        return csr_matrix(spectrum)
    return np.array(spectrum)
############################################################################
############################################################################
############################################################################


def gen_kernel_matrix(dataset,create_file):
    if dataset=='MNIST':
        bandwidth = 0.1
        data = MNIST('datasets/MNIST')
        train_x, train_y = data.load_training()
        test_x, test_y = data.load_testing()
        train_x = np.array(train_x) / 255
        train_y = np.array(train_y)
        test_x = np.array(test_x) / 255
        test_y = np.array(test_y)
        rbf = RBF(math.sqrt(1/(2*bandwidth)))
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = np.array(random.sample(train_x.tolist(),num_landmark))
        landmark_kernel_matrix = rbf(landmarks)
        train_kernel_matrix = rbf(train_x,landmarks)
        test_kernel_matrix = rbf(test_x,landmarks)
        save_list = {'dataset':'MNIST','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/MNIST.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===MNIST Done===')
        return save_list

    if dataset=='FashionMNIST':
        bandwidth = 0.01
        data = MNIST('datasets/FashionMNIST')
        train_x, train_y = data.load_training()
        test_x, test_y = data.load_testing()
        train_x = np.array(train_x) / 255
        train_y = np.array(train_y)
        test_x = np.array(test_x) / 255
        test_y = np.array(test_y)
        rbf = RBF(math.sqrt(1/(2*bandwidth)))
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = np.array(random.sample(train_x.tolist(),num_landmark))
        landmark_kernel_matrix = rbf(landmarks)
        train_kernel_matrix = rbf(train_x,landmarks)
        test_kernel_matrix = rbf(test_x,landmarks)
        save_list = {'dataset':'MNIST','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/FashionMNIST.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===FashionMNIST Done===')
        return save_list

    if dataset=='NCI1':
        NCI1 = fetch_dataset("NCI1",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = NCI1.data, NCI1.target
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = Propagation(t_max=5)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'NCI1','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/NCI1.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===NCI1 Done===')
        return save_list

    if dataset=='ENZYMES':
        ENZYMES = fetch_dataset("ENZYMES",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = ENZYMES.data, ENZYMES.target
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        train_y = train_y-1
        test_y = test_y-1
        gk = PropagationAttr(t_max=1)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'ENZYMES','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/ENZYMES.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===ENZYMES Done===')
        return save_list
    

    if dataset=='Protein':
        class0_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Actinorhodopsin.txt"
        class1_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Bacteriorhodopsin.txt"
        class2_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Halorhodopsin.txt"
        class3_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Proteorhodopsin.txt"
        class4_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Sensory-rhodopsin.txt"
        class5_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Xanthorhodopsin.txt"
        files = [class0_file,class1_file,class2_file,class3_file,class4_file,class5_file]
        sequences = []
        labels = []
        for num in range(len(files)):
            file = files[num]
            io = open(file, 'r')
            Lines = io.readlines()
            for line in Lines:
                seq = line.strip().split("::")
                if len(seq) != 2:
                    continue
                seq = seq[1]
                sequences.append(seq)
                labels.append(num)
        train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=42)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = random.sample(train_x,num_landmark)
        landmark_feature = gappypair_kernel(landmarks, k=2, g=1, t=2, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        landmark_kernel_matrix = np.matmul(landmark_feature, np.transpose(landmark_feature))
        train_feature = gappypair_kernel(train_x, k=2, g=1, t=2, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        train_kernel_matrix = np.matmul(landmark_feature,train_feature.T).T
        test_feature = gappypair_kernel(test_x, k=2, g=1, t=2, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        test_kernel_matrix = np.matmul(landmark_feature,test_feature.T).T
        save_list = {'dataset':'Protein','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/Protein.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===Protein Done===')
        return save_list


    if dataset=='SMS':
        files = ["./datasets/SMS/SMSSpamCollection"]
        sequences = []
        labels = []
        for num in range(len(files)):
            file = files[num]
            io = open(file, 'r')
            Lines = io.readlines()
            for line in Lines:
                seq = line.strip().split("\t")
                if len(seq) != 2:
                    continue
                seq[1] = re.sub('[^A-Za-z0-9]+', '', seq[1]).upper()
                sequences.append(seq[1])
                labels.append(1 if seq[0] == 'ham' else 0)
        train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=42)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = random.sample(train_x,num_landmark)
        landmark_feature = gappypair_kernel(landmarks, k=1, g=1, t=3, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        landmark_kernel_matrix = np.matmul(landmark_feature, np.transpose(landmark_feature))
        train_feature = gappypair_kernel(train_x, k=1, g=1, t=3, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        train_kernel_matrix = np.matmul(landmark_feature,train_feature.T).T
        test_feature = gappypair_kernel(test_x, k=1, g=1, t=3, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        test_kernel_matrix = np.matmul(landmark_feature,test_feature.T).T
        save_list = {'dataset':'SMS','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/SMS.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===SMS Done===')
        return save_list


    if dataset=='MUTAG':
        MUTAG = fetch_dataset("MUTAG",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = MUTAG.data, MUTAG.target
        y[y==-1] = 0
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = Propagation(t_max=10)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'MUTAG','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/MUTAG.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===MUTAG Done===')
        return save_list
    

    if dataset=='DD':
        DD = fetch_dataset("DD",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = DD.data, DD.target
        y[y==2] = 0
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = Propagation(t_max=2)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'DD','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/DD.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===DD Done===')
        return save_list
    

    if dataset=='BZR':
        BZR = fetch_dataset("BZR",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = BZR.data, BZR.target
        y[y==-1] = 0
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = PropagationAttr(t_max=1)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'BZR','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/BZR.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===BZR Done===')
        return save_list
    

    if dataset=='COX2':
        COX2 = fetch_dataset("COX2",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = COX2.data, COX2.target
        y[y==-1] = 0
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = PropagationAttr(t_max=5)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'COX2','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/COX2.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===COX2 Done===')
        return save_list
    


    if dataset=='Mutagenicity':
        Mutagenicity = fetch_dataset("Mutagenicity",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
        G, y = Mutagenicity.data, Mutagenicity.target
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = Propagation(t_max=3)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'Mutagenicity','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/Mutagenicity.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===Mutagenicity Done===')
        return save_list


    if dataset=='NCI109':
        NCI109 = fetch_dataset("NCI109",verbose=True,download_if_missing=True,prefer_attr_nodes=True)
        G, y = NCI109.data, NCI109.target
        train_x, test_x, train_y, test_y = train_test_split(G, y, test_size=0.2, random_state=42)
        gk = Propagation(t_max=5)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = np.array(random.sample(train_x,num_landmark))
        landmark_kernel_matrix = gk.fit_transform(landmarks)
        train_kernel_matrix = gk.transform(train_x)
        test_kernel_matrix = gk.transform(test_x)
        save_list = {'dataset':'NCI109','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/NCI109.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===NCI109 Done===')
        return save_list


    if dataset=='splice':
        file = "./datasets/splice/splice.data"
        sequences = []
        labels = []
        io = open(file, 'r')
        Lines = io.readlines()
        for line in Lines:
            parse = line.strip().split(",")
            parse[2] = re.sub('[^A-Za-z0-9]+', '', parse[2]).upper()
            sequences.append(parse[2])
            if parse[0] == 'EI':
                labels.append(0)
            elif parse[0] == 'IE':
                labels.append(1)
            elif parse[0] == 'N':
                labels.append(2)
            else:
                print('Error')
                print(parse[0])
                exit()
        train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=42)
        num_landmark = max(int(len(train_x) * 0.02),300)
        landmarks = random.sample(train_x,num_landmark)
        landmark_feature = gappypair_kernel(landmarks, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        landmark_kernel_matrix = np.matmul(landmark_feature, np.transpose(landmark_feature))
        train_feature = gappypair_kernel(train_x, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        train_kernel_matrix = np.matmul(landmark_feature,train_feature.T).T
        test_feature = gappypair_kernel(test_x, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        test_kernel_matrix = np.matmul(landmark_feature,test_feature.T).T
        save_list = {'dataset':'splice','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/splice.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===splice Done===')
        return save_list




    if dataset=='promoter':
        file = "./datasets/promoter/promoters.data"
        sequences = []
        labels = []
        io = open(file, 'r')
        Lines = io.readlines()
        for line in Lines:
            parse = line.strip().split(",")
            parse[2] = re.sub('[^A-Za-z0-9]+', '', parse[2]).upper()
            sequences.append(parse[2])
            if parse[0] == '+':
                labels.append(0)
            elif parse[0] == '-':
                labels.append(1)
            else:
                print('Error')
                print(parse[0])
                exit()
        train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=42)
        num_landmark = min(max(int(len(train_x) * 0.02),300),len(train_x))
        landmarks = random.sample(train_x,num_landmark)
        landmark_feature = gappypair_kernel(landmarks, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        landmark_kernel_matrix = np.matmul(landmark_feature, np.transpose(landmark_feature))
        train_feature = gappypair_kernel(train_x, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        train_kernel_matrix = np.matmul(landmark_feature,train_feature.T).T
        test_feature = gappypair_kernel(test_x, k=2, g=1, t=0, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        test_kernel_matrix = np.matmul(landmark_feature,test_feature.T).T
        save_list = {'dataset':'promoter','num_landmark':num_landmark,'landmark_kernel_matrix':landmark_kernel_matrix,'train_kernel_matrix':train_kernel_matrix,
                     'test_kernel_matrix':test_kernel_matrix,'train_label':train_y,'test_label':test_y,'train_num':len(train_y),'test_num':len(test_y),
                     'landmarks':landmarks}
        if create_file==True:
            with open('KernelMatrices/promoter.pkl', 'wb') as output:
                pickle.dump(save_list, output, pickle.HIGHEST_PROTOCOL)
        print(len(train_y))
        print(len(test_y))
        print(num_landmark)
        print(landmark_kernel_matrix.shape)
        print(train_kernel_matrix.shape)
        print(test_kernel_matrix.shape)
        print('===promoter Done===')
        return save_list






if __name__ == "__main__":
    #gen_kernel_matrix("SMS",True)
    #gen_kernel_matrix("Protein",True)
    #gen_kernel_matrix("ENZYMES",True)
    #gen_kernel_matrix("NCI1",True)
    #gen_kernel_matrix("FashionMNIST",True)
    #gen_kernel_matrix("MNIST",True)
    #gen_kernel_matrix("MUTAG",True)
    #gen_kernel_matrix("DD",True)
    #gen_kernel_matrix("BZR",True)
    #gen_kernel_matrix("COX2",True)
    #gen_kernel_matrix("Mutagenicity",True)
    #gen_kernel_matrix("NCI109",True)
    #gen_kernel_matrix("splice",True)
    #gen_kernel_matrix("promoter",True)
    pass

