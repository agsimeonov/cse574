""" Some useful routines to understand programming assignment one """
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random

"""
The data is provided as a MATLAB binary which can be loaded into Python using 
the following command
"""
mat = loadmat('../mnist_all.mat')

"""
The data is essentially a dictionary containing several matrices (train0-train9 
are training data matrices and test0-test9 are the test data matrices)
"""
print mat.keys()

"""
Each matrix in the dictionary corresponds to a digit. Each row is one image
(28 x 28 matrix flattened into a 784 length vector)
"""
# For example train9 is all training images corresponding to the digit 9
train9 = mat.get('train9')
print train9.shape

"""
To view any one digit you need to take a row and reshape it back as a matrix
"""
# plot 100 random images from the digit '9' data set
s = random.sample(range(train9.shape[0]),100)
fig = plt.figure(figsize=(12,12))
for i in range(100):
    plt.subplot(10,10,i)
    row = train9[s[i],:]
    # note that each row is a flattened image
    # we first reshape it to a 28x28 matrix
    plt.imshow(np.reshape(row,((28,28))))
    plt.axis('off')

""" Here are random 10 images for each digit """
fig = plt.figure(figsize=(12,12))
for i in range(10):
    trainx = mat.get('train'+str(i))
    # note the use of function random.sample to extract 10 rows randomly
    s = random.sample(range(trainx.shape[0]),10)
    for j in range(10):
        plt.subplot(10,10,10*i+j+1)
        row = trainx[s[i],:]
        # note that each row is a flattened image
        # we first reshape it to a 28x28 matrix
        plt.imshow(np.reshape(row,((28,28))))
        plt.axis('off')

"""
To check the total number of training examples you need to iterate over all
trainx matrices
"""
trainsize = 0
for i in range(10):
    m = mat.get('train'+str(i))
    trainsize = trainsize + m.shape[0]
print trainsize

"""
Pointers about splitting a matrix into two random parts
"""
# Assume we have a matrix (I will repurpose one of the trainx matrices from above)
A = mat.get('train0')
print A.shape

# I want to break it into two random parts, one matrix with 1000 rows and second with the rest
a = range(A.shape[0])
aperm = np.random.permutation(a)
A1 = A[aperm[0:1000],:]
A2 = A[aperm[1000:],:]
print A1.shape
print A2.shape

""" Display plots """
plt.show()
