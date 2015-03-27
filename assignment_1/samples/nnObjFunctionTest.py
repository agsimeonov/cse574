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
        delta_hidden = (1.0-hidden_output) * hidden_output * sum_delta_w
        gradient_w1 += delta_hidden.reshape((n_hidden,1)) * np.hstack((example,np.ones(1.0)))
        
    # At this point, we average error and two gradients
    error /= -NUM_EXAMPLES
    
    # Regularization error
    sum_squares_weight = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    error += (lambdaval * sum_squares_weight) / (2.0*NUM_EXAMPLES)
    
    # Regularization gradients
    gradient_w1 = (gradient_w1 + lambdaval * w1) / NUM_EXAMPLES
    gradient_w2 = (gradient_w2 + lambdaval * w2) / NUM_EXAMPLES
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()),0)
    obj_val = error
    
    return (obj_val,obj_grad)


"""**************Neural Network Script Starts here********************************"""

n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)

obj_val, obj_grad = nnObjFunction(params, n_input, n_hidden, n_class, training_data, training_label, lambdaval)

print obj_val
print obj_grad

# Pickle tests below this line
import pickle

opts = {'maxiter' : 100}
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
nn_params = minimize(nnObjFunction, params, jac=True, args=args,method='CG', options=opts)
    
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

pickle_name = str(n_hidden) + "hidden_" + str(lambdaval) + "lambda.pickle"
pickle.dump((n_hidden, w1, w2, lambdaval), open(pickle_name, "wb"))
print pickle.load(open(pickle_name, "rb"))
