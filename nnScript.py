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

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return 1.0 / (1.0 + np.exp(-1.0 * z))
    
    

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
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    n_train = training_data.shape[0]


    regularization_term = np.sum(w1 * w1) + np.sum(w2 * w2)
    regularization_term = (lambdaval / (2 * n_train)) * regularization_term

    (predicted_labels, hidden_outputs) = nnPredict(w1, w2, training_data)

    real_output_vector = zeros((1, n_class))
    predicted_output_vector = zeros((1, n_class))
    obj_val = 0.0
    for n in range(n_train):
        real_output_vector[training_label[n]] = 1
        predicted_output_vector[predicted_labels[n]] = 1
	    
        obj_val = obj_val + np.sum(predicted_output_vector * np.log(real_output_vector) + ((1 - predicted_output_vector) * np.log(1 - real_output_vector)))
        real_output_vector[training_label[n]] = 0
        predicted_output_vector[predicted_labels[n]] = 0

    obj_val = -1 / n_train * obj_val
    obj_val = obj_val + regularization_term

    update_hidden_weights = zeros((n_hidden, n_input + 1)); 
    for k in range(n_train): 
        real_output_vector[training_label[k]] = 1
        predicted_output_vector[predicted_labels[k]] = 1
        for i in range(n_hidden):
      	    gradient1 = np.sum((predicted_output_vector - real_output_vector) * w2[:, i])
	    for j in range(n_input + 1):
	        gradient2 = (1 - hidden_outputs[k, i]) * hidden_outputs[k, i] * training_data[k, j]
	        update_hidden_weights[i, j] = update_hidden_weights[i, j] + gradient + gradient2
	real_output_vector[training_label[k]] = 0
	predicted_output_vector[predicted_labels[k]] = 0
    w1 = (1 / n_train) * (update_hidden_weights + (lambdaval * w1)) 

    update_output_weights = zeros((n_class, n_hidden + 1)); 
    for k in range(n_train): 
        real_output_vector[training_label[k]] = 1
        predicted_output_vector[predicted_labels[k]] = 1
        for i in range(n_class):
            for j in range(n_hidden + 1):
		gradient = (predicted_output_vector[i] - real_output_vector[i]) * hidden_outputs[k, j]
		update_hidden_weights[i, j] = update_hidden_weights[i, j] + gradient 
	real_output_vector[training_label[k]] = 0
	predicted_output_vector[predicted_labels[k]] = 0
    w2 = (1 / n_train) * (update_hidden_weights + (lambdaval * w2))
    #Your code here
    #
    #
    #
    #
    #
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
