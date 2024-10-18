from NysHD_train import *
import time
import numpy as np





def run_exp(precomputed_kernel):
    accs = {}
    f1s = {}
    times = {}
    lr = {'NCI1':0.1,'ENZYMES':0.1,'Protein':1,'SMS':1,'MNIST':1,'FashionMNIST':1,'MUTAG':0.1,'NCI109':0.1,'DD':0.1,'BZR':0.1,'COX2':0.1,'Mutagenicity':0.1,'splice':1,'promoter':1}
    for dataset in ['NCI1','promoter','splice','Protein','SMS','ENZYMES','MUTAG','NCI109','DD','BZR','COX2','Mutagenicity']:
        accs[dataset] = []
        f1s[dataset] = []
        times[dataset] = []
        for trial in range(10):
            now = time.time()
            model = hd_model(dataset,10000,lr[dataset],20,precomputed_kernel)
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
    if precomputed_kernel:
        with open('results_precomputed_kernel.txt', 'w') as f:
            print(accs, file=f)
            print("========================", file=f)
            print(f1s, file=f)
            print("========================", file=f)
            print(times, file=f)
            print("========================", file=f)
    else:
        with open('results_compute_kernel.txt', 'w') as f:
            print(accs, file=f)
            print("========================", file=f)
            print(f1s, file=f)
            print("========================", file=f)
            print(times, file=f)
            print("========================", file=f) 
    print("===Exp End===")



if __name__ == "__main__":
    #run_exp(True)
    run_exp(False)































