import numpy as np
import matplotlib.pyplot as plt
import csv


arr = np.loadtxt("result.csv",delimiter=",", dtype=float)

mean = np.mean(arr,axis=1)
std = np.std(arr,axis=1)
x = np.array([5,10,15,20,25,30,35,40])

print(mean)
print(std)

plt.rcParams.update({'font.size': 22})
plt.plot(x, mean)
plt.grid()
plt.errorbar(x, mean,yerr = std, fmt ='o')
#plt.xlabel('Landmark Percentage %') 
#plt.ylabel('Spectral Norm') 
#plt.title('Approximation Error')
plt.savefig('plot')
