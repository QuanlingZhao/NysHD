from train import *
import time
import numpy as np





def run_exp():
    accs = {}
    f1s = {}
    times = {}
    lr = {'NCI1':1,'ENZYMES':0.2,'Protein':1,'SMS':1,'MNIST-Linear':1,'FashionMNIST-Linear':1,'MNIST-Nonlinear':1,'FashionMNIST-Nonlinear':1,'MUTAG':1,'NCI109':1,'DD':1,'BZR':1,'COX2':1,'Mutagenicity':1,'splice':1,'promoter':1}
    for dataset in ['splice','promoter','Protein','SMS','ENZYMES','NCI1','MUTAG','NCI109','DD','BZR','COX2','Mutagenicity']:    
        accs[dataset] = []
        f1s[dataset] = []
        times[dataset] = []
        for trial in range(10):
            now = time.time()
            model = hd_model(dataset,lr[dataset],20)
            model.train()
            acc, f1 = model.test()
            elapsed = int(time.time() - now)
            accs[dataset].append(acc)
            f1s[dataset].append(f1)
            times[dataset].append(elapsed)
            print("Elapsed:", elapsed)
            print(' ')
        accs[dataset].append(np.mean(accs[dataset]))
        accs[dataset].append(np.std(accs[dataset][:-1]))
        f1s[dataset].append(np.mean(f1s[dataset]))
        f1s[dataset].append(np.std(f1s[dataset][:-1]))
        times[dataset].append(np.mean(times[dataset]))
        times[dataset].append(np.std( times[dataset][:-1]))

    with open('hd_results.txt', 'w') as f:
        print(accs, file=f)
        print("========================", file=f)
        print(f1s, file=f)
        print("========================", file=f)
        print(times, file=f)
        print("========================", file=f) 
    print("===Exp End===")



if __name__ == "__main__":
    run_exp()































