import numpy as np
import scipy as sp
import scipy.linalg as sl

a = [1,2,3,4,5,1,1,12,3,4,5,6]
b = np.array(a)
print(b<5)
print(b[b<5])