import numpy as np
import matplotlib.pyplot as plt
import csv


arr = np.loadtxt("result.csv",delimiter=",", dtype=float)

mean = np.mean(arr,axis=1) * 100
std = np.std(arr,axis=1) * 100
x = np.array([1,3,5,7,9])

print(mean)
print(std)

plt.rcParams.update({'font.size': 22})
plt.plot(x, mean)
plt.grid()
plt.errorbar(x, mean,yerr = std, fmt ='o')
plt.xlabel('Landmark Percentage %') 
plt.ylabel('Classification Accuracy %') 
plt.title('NCI1')
plt.tight_layout()
plt.savefig('plot')
