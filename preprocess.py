import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def stack(mat, name, digits):
    stack = None
    for digit in digits:
        key = name + str(digit)
        if stack is None:
            stack = mat.get(key)
        else:
            stack = np.concatenate((stack, mat.get(key)))
    return stack

def label(data, digits):
    label = []
    for i in range(data.shape[0]):
        label.append(int(i/(data.shape[0]/len(digits))))
    label = np.array(label)
    return label[:,np.newaxis]

def normalize(data):
    data = data.astype(np.float32)
    return data / 255

def preprocess():
    """ Input:
        Although this function doesn't have any input, you are required to load
        the MNIST data set from file 'mnist_all.mat'.
        
        Output:
        train_data: matrix of training set. Each row of train_data contains
        feature vector of a image
        train_label: vector of label corresponding to each image in the training
        set
        validation_data: matrix of training set. Each row of validation_data
        contains feature vector of a image
        validation_label: vector of label corresponding to each image in the
        training set
        test_data: matrix of training set. Each row of test_data contains
        feature vector of a image
        test_label: vector of label corresponding to each image in the testing
        set
        
        Some suggestions for preprocessing step:
        - divide the original data set to training, validation and testing set
        with corresponding labels
        - convert original data set from integer to double by using double()
        function
        - normalize the data to [0, 1]
        - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    
    #Your code here
    TRAIN_NAME = 'train'
    TEST_NAME = 'test'
    DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 1.1 Stack all training matrices into one 60000 x 784 matrix
    train_data = stack(mat, TRAIN_NAME, DIGITS)
    # 1.2 Do the same for test matrices
    test_data = stack(mat, TEST_NAME, DIGITS)
    
    # 2.1 Create a 60000 length vector with true labels (digits) for each
    # training example
    train_label = label(train_data, DIGITS)
    # 2.2 Same for test data
    test_label = label(test_data, DIGITS)
    
    # 3 Normalize the training matrix and test matrix so that the values are
    # between 0 and 1
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    
    # 4 Randomly split the 60000 x 784 normalized matrix into two matrices:
    # training matrix (50000 x 784) and validation matrix (10000 x 784). Make
    # sure you split the true labels vector into two parts as well.
    train = np.concatenate((train_label, train_data), 1)
    train = np.random.permutation(train)
    train = np.hsplit(train, [1])
    train_label = train[0]
    train_data = train[1]
    train_label = np.split(train_label, [50000])
    train_data = np.split(train_data, [50000])
    validation_label = train_label[1]
    validation_label = validation_label.astype(int)
    validation_data = train_data[1]
    train_label = train_label[0]
    train_label = train_label.astype(int)
    train_data = train_data[0]
    
    # 5 Feature selection
    ranges = np.ptp(train_data, axis=0)
    i = 0
    delete_indexes = []
    for x in ranges:
        if x == 0.0:
            delete_indexes.append(i)
        i = i + 1
    train_data = np.delete(train_data, delete_indexes, axis=1)

#    print train_data.shape
#
#    a = np.arange(6).reshape(2,3)
#    print np.ptp(a, axis=0)
#    print np.ptp(a, axis=1)
#    print a
#    a = np.delete(a, [0], axis=1)
#    print a

#    a = np.arange(6).reshape(2,3)
#    print a
#    print np.ptp(a, axis=0)
#    print np.ptp(a, axis=1)

#    print train_label.shape
#    print train_data.shape
#    print validation_label.shape
#    print validation_data.shape
#    print type(validation_label)

   
#    a = np.arange(6).reshape(2,3)
#    print a
#    aperm = np.random.permutation(a)
#    print aperm
#    a = a.astype(np.float32)
#    a = a / 5.0
#    print a
#    a = 1.0 / (1.0 + np.exp(-1.0 * a))
#    print a
    
    
#    from random import shuffle
#    indexes = [[i] for i in range(len(train_data))]
#    shuffle(indexes)
#    training_indexes = indexes[0:50000]
#    validation_indexes = indexes[50000:60000]
#    # 4.1 Training matrix (50000 x 784) and validation matrix (10000 x 784)
#    validation_data = indexSplit(train_data, validation_indexes)
#    train_data = indexSplit(train_data, training_indexes)
#    # 4.2 Make sure you split the true labels vector into two parts as well
#    validation_label = indexSplit(train_label, validation_indexes)
#    train_label = indexSplit(train_label, training_indexes)

#    arr = np.arange(9).reshape((3, 3))
#    print arr
#    np.random.shuffle(arr)
#    print arr
#    jojo = [55,66,77]
#    jojo = np.array(jojo)
#    jojo = jojo[:,np.newaxis]
#    print jojo
#    jojo = np.concatenate((jojo, arr), 1)
#    print jojo
#    splito = np.hsplit(jojo, [1])
#    print splito

#    stackyyz = stack(train_label, training_indexes)
#    for x in stackyyz:
#        print x
#    
#    print len(training_indexes)
#    print len(validation_indexes)
#    print DIGITS[5:9]

#    x = [[i] for i in range(10)]
#    shuffle(x)
#    print x
#    print shuffle(DIGITS)
#    a = np.arange(6).reshape(2,3)
#    print a
#    print a[0]
#    print a[1]
#    for x in a:
#        print x

#    A = np.array(np.arange(1,10))
#    B = A.reshape((3,3))*255
#    print B.astype(np.float32)
#    A = np.array([1,2,3,4,5,6,7,8,9])
#    B = A[:,np.newaxis]
#    print A
#    print B
#    for key in mat:
#         print key
#        if 'train' in key:
#            print type(mat.get(key))/
#            print key

    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

preprocess()
