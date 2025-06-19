import numpy as np
import math

W1 = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1 = np.array( [-9.82, -9.28,  0.96] )
W2 = np.array( [[-31.18], [-27.59], [-32.56]] )
b2 = np.array( [15.41] ) 

#sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


#dense dunction
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w=W[:, j]
        z=np.dot(a_in, w) + b[j]
        a_out[j] = sigmoid(z) 
    return a_out

#sequential dunction
def sequential(x):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)
    f_x = a2
    return f_x

print(sequential())
