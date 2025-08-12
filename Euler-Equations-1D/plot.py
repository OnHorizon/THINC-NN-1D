import numpy as np 
import matplotlib.pyplot as plt 

data = np.loadtxt('sol.csv',delimiter=',',skiprows=1)

plt.plot(data[:,0],data[:,1],label='THINC-NN SSPRK3')
plt.grid()
plt.legend()
plt.savefig('plot.png',dpi=500)
