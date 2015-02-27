import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


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
    
    # Initialize variables
    train_data = np.zeros((0,784))
    test_data = np.zeros((0,784))
    train_label = np.zeros((0,))
    test_label = np.zeros(0,)
    
    # For every digit, stack corresponding matrices and labels
    for i in range(0,10):
        ''' Stack vertically training and test matrices '''
        train_data = np.vstack((train_data, mat['train'+str(i)]))
        test_data = np.vstack((test_data, mat['test'+str(i)]))
        
        ''' Create and stack label vectors '''
        train_label = np.hstack((train_label, i * np.ones(mat['train'+str(i)].shape[0])))
        test_label = np.hstack((test_label, i * np.ones(mat['test'+str(i)].shape[0])))
    
    
    # Normalize training and test matrices
    train_data = np.double(train_data)/255
    test_data = np.double(test_data)/255
    
    # Randomly split the training matrix in
    # training (50000 x 784) and validation (10000 x 784).
    # Label vector gets split accordingly
    ''' Make train_label a column vector '''
    train_label = train_label.reshape((60000,1))
    
    ''' Shuffle the rows '''
    T = np.random.permutation(np.hstack((train_data, train_label)))
    
    ''' First 50000 rows are the new training data, the rest is validation data '''
    train = T[:50000]
    valid = T[50000:]
    
    ''' Last column is training labels '''
    train_data = train[:,:-1]
    train_label = train[:,-1:]
    
    validation_data = valid[:,:-1]
    validation_label = valid[:,-1:]
    
    # Feature selection - eliminate undistinguishing attributes
    delete_indexes = []
    for i in range(784):
        if np.ptp(train_data[:,i]) == 0.0 and \
            np.ptp(validation_data[:,i]) == 0.0 and\
            np.ptp(test_data[:,i]) == 0.0:
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
    for 
    
    
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
    
    labels = []
    num_examples = data.shape[0]
    
    # Add attribute d + 1 - column of 1's
    data = np.hstack((data, np.ones(num_examples).reshape((num_examples,1))))
    
    # For every example, compute the predicted digit
    # and append it to the label list
    for example in data:
        output_hidden_nodes = sigmoid(np.sum(w1*example, axis=1))
        output = sigmoid(np.sum(w2*output_hidden_nodes, axis=1))
        predicted_digit = np.argmax(output)[0]
        labels.append(predicted_digit)
    
    # Return an column vector    
    return np.array(labels).reshape((num_examples,1))
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 4;
				   
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
