import numpy as np

MAX_SEQ_LEN = 102
a = np.ones([14,300])
print(MAX_SEQ_LEN-a.shape[0])

a = np.pad(a,[(0,MAX_SEQ_LEN-a.shape[0]),(0,0)], mode='constant', constant_values=0)

print(a)
print(a.shape)
