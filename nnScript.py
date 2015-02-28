import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import strftime, localtime
import sys

def perceptron(weights, example):
    '''
    # It computes the ouput of a layer of perceptrons
    # A single output is = sigmoid(linear combination of input attributes)
    #
    # weights : matrix of weights, size m x d+1
    # weights(i,j) : weight from feature j to unit i
    # example : vector of attributes, size d
    '''
    linear_comb = np.sum(weights*(np.hstack((example,np.ones(1.0)))), axis=1)
    output = sigmoid(linear_comb)
    
    return output

def feed_forward(w1, w2, example):
    '''
    # It computes the ouputs of a two layer network of perceptrons
    #
    # w1 : matrix of weights, size m x d+1
    # w2 : matrix of weights, size k x d+1
    # weights(i,j) : weight from feature j to unit i
    # example : vector of attributes, size d
    '''
    output_hidden = perceptron(w1, example)
    output = perceptron(w2, output_hidden)
    
    return (output_hidden, output)
    
    
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
    
    # Initialize variables and constants
    NUM_TRAIN = 50000
    NUM_ATTR = 784
    training_data = np.zeros((0,NUM_ATTR))
    test_data = np.zeros((0,NUM_ATTR))
    training_label = np.zeros((0,))
    test_label = np.zeros(0,)
    
    # For every digit, stack corresponding matrices and labels
    for i in range(0,10):
        ''' Stack vertically training and test matrices '''
        training_data = np.vstack((training_data, mat['train'+str(i)]))
        test_data = np.vstack((test_data, mat['test'+str(i)]))
        
        ''' Create and stack label vectors '''
        training_label = np.hstack((training_label, i * np.ones(mat['train'+str(i)].shape[0])))
        test_label = np.hstack((test_label, i * np.ones(mat['test'+str(i)].shape[0])))
        
    # Normalize training and test matrices
    training_data = training_data.astype(np.float64, copy=False)/255
    test_data = test_data.astype(np.float64, copy=False)/255
    
    # Randomly split the training matrix in
    # training (50000 x 784) and validation (10000 x 784).
    # Label vector gets split accordingly
    ''' Create a permutation of the rows '''
    perm = np.random.permutation(range(training_data.shape[0]))
    
    ''' Extract rows using the permutation '''
    train_data = training_data[perm[0:NUM_TRAIN],:]
    validation_data = training_data[perm[NUM_TRAIN:],:]
    
    ''' Do the same for label vector '''
    train_label = training_label[perm[0:NUM_TRAIN]]
    validation_label = training_label[perm[NUM_TRAIN:]]
    
    # Feature selection - eliminate undistinguishing attributes
    delete_indexes = []
    for i in range(NUM_ATTR):
        if np.ptp(train_data[:,i]) == 0.0 and \
            np.ptp(validation_data[:,i]) == 0.0:
                delete_indexes.append(i)

    train_data = np.delete(train_data, delete_indexes, axis=1)
    validation_data = np.delete(validation_data, delete_indexes, axis=1)
    test_data = np.delete(test_data, delete_indexes, axis=1)
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
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
    
    NUM_EXAMPLES = training_data.shape[0]
    
    # For each example, compute error and gradient
    # For simplicity, there will be two gradient matrices:
    # one for the input weights, one for the output weights
    #
    # Accumulators for error and two gradients
    error = 0.0
    gradient_w1 = 0.0
    gradient_w2 = 0.0
    
    for i in range(NUM_EXAMPLES):
        example = training_data[i]
        ''' Create correct output vector setting to 1.0 the entry 
            correspondent to actual label'''
        correct_output = np.zeros(n_class)
        correct_output[training_label[i]] = 1.0
        
        ''' Perform a feed forward pass to get outputs '''
        hidden_output, output = feed_forward(w1,w2,example)
        
        ''' Error on example i '''
        error += np.sum(correct_output * np.log(output) + (1.0-correct_output) * np.log(1.0-output))
        
        # Gradient calculation
        ''' delta and gradient w2 '''
        delta_output = output - correct_output
        gradient_w2 += delta_output.reshape((n_class,1)) * np.hstack((hidden_output,np.ones(1.0)))
        
        ''' gradient w1 '''
        sum_delta_w = np.dot(delta_output,w2) # it has one entry per hidden node + 1
        sum_delta_w = sum_delta_w[:-1] # we get rid of the last entry
        delta_hidden = (1-hidden_output) * hidden_output * sum_delta_w
        gradient_w1 += delta_hidden.reshape((n_hidden,1)) * np.hstack((example,np.ones(1.0)))
        
    # At this point, we average error and two gradients
    error /= -NUM_EXAMPLES
    gradient_w1 /= NUM_EXAMPLES
    gradient_w2 /= NUM_EXAMPLES
    
    # Regularization error
    sum_squares_weight = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    error += (lambdaval / (2.0*NUM_EXAMPLES)) * sum_squares_weight
    
    # Regularization gradients
    gradient_w1 += (lambdaval / NUM_EXAMPLES) * w1
    gradient_w2 += (lambdaval / NUM_EXAMPLES) * w2
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()),0)
    obj_val = error
    
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
    
    labels = []
    
    # For every example, compute the predicted digit
    # and append it to the label list
    for example in data:
        ''' Compute output network '''
        (output_hidden, output) = feed_forward(w1,w2,example)
        ''' Maximum value among output units is predicted digit '''
        predicted_digit = np.argmax(output)
        labels.append(predicted_digit)
    
    # Return a vector with labels   
    return np.array(labels)
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = int(sys.argv[1]);
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = float(sys.argv[2]);

# Print training parameters
print('Training neural network with parameters: hidden nodes = ' + str(n_hidden) + ' lambda = ' + str(lambdaval))

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

# Print time - begin training
print('\nBeginning training. Time: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

# Print time - end training
print('\nEnding training. Time: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
print('\n\n\n')