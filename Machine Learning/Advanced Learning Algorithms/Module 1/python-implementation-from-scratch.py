import numpy as np
import math
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
    a3 = dense(a2,W3,b3)
    a4 = dense(a3,W4,b4)
    f_x = a4
    return f_x
