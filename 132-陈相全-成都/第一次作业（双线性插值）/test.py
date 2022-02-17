# import numpy as np
# a = int(np.floor(-0.1))
# print(a)

b = [2,1,4,3,4,2,3]
result = b[0]
for i in range(1,len(b)):
    result = result^b[i]
print(result)