import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    numClass = int(np.max(y))
    d = np.shape(X)[1]
    means = np.empty((d, numClass));

    covmat = np.zeros(d)
    for i in range (1, numClass + 1):
        c = np.where(y==i)[0]
        trainData = X[c,:]
        means[:, i-1] = np.mean(trainData, axis=0).transpose()
        covmat = covmat + (np.shape(trainData)[0]-1) * np.cov(np.transpose(trainData))


    covmat = (1.0/(np.shape(X)[0] - numClass)) * covmat;# - numClass));

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    numClass = int(np.max(y))
    d = np.shape(X)[1]
    means = np.empty((d, numClass));

    covmats = []
    for i in range (1, numClass+1):
        c = np.where(y==i)[0]
        trainData = X[c,:]
        means[:, i-1] = np.mean(trainData, axis=0).transpose()
        covmats.append(np.cov(np.transpose(trainData)))

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    N = np.shape(Xtest)[0];
    #d = np.shape(Xtest)[1];
    classCount = np.shape(means)[1];
    #normalize = 1/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmat),1/2));
    countCorrect = 0.0;
    invCov = np.linalg.inv(covmat);
    ytest = ytest.astype(int);
    for i in range (1, N + 1):
        pdf = 0;
        classNum = 0;
        testX = np.transpose(Xtest[i-1,:]);
        for k in range (1, classCount+1):
            result = np.exp((-1/2)*np.dot(np.dot(np.transpose(testX - means[:, k-1]),invCov),(testX - means[:, k-1])));
            if (result > pdf):
                classNum = k;
                pdf = result;
        if (classNum == ytest[i-1]):
            countCorrect = countCorrect + 1;
    acc = countCorrect/N;
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuray value
    classCount = np.shape(means)[1];
    normalizers = np.zeros(classCount);
    for i in range (1, classCount+1):
        d = np.shape(covmats[i-1])[0];
        normalizers[i-1] = 1.0/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmats[i-1]),1/2));

        covmats[i-1] = np.linalg.inv(covmats[i-1]);
    N = np.shape(Xtest)[0];
    #d = np.shape(Xtest)[1];
    #normalize = 1/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmat),1/2));
    countCorrect = 0.0;
    ytest = ytest.astype(int);
    for i in range (1, N + 1):
        pdf = 0;
        classNum = 0;
        testX = np.transpose(Xtest[i-1,:]);
        for k in range (1, classCount+1):
            invCov = covmats[k-1];
            result = normalizers[k-1]*np.exp((-1/2)*np.dot(np.dot(np.transpose(testX - means[:, k-1]),invCov),(testX - means[:, k-1])));
            if (result > pdf):
                classNum = k;
                pdf = result;
        if (classNum == ytest[i-1]):
            countCorrect = countCorrect + 1;
    acc = countCorrect/N;
    return acc

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    w = np.dot(np.linalg.pinv(X), y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    w = (X.shape[0] * lambd * np.identity(X.shape[1])) + np.dot(X.T, X)
    w = np.linalg.inv(w)
    w = np.dot(w, X.T)
    w = np.dot(w, y)
    return w

def squaredSum(w, X, y):
    squaredSum = 0
    for i in range(0, X.shape[0]):
        squaredSum += np.square(y[i] - np.dot(w.T, X[1,:]))
    return squaredSum

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    rmse = (1.0/X.shape[0]) * np.sqrt(squaredSum(w, Xtest, ytest))
    return rmse[0]

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    error = ((1.0/(2.0 * X.shape[0])) * squaredSum(w, X, y)) + (.5 * lambd * np.dot(w.T, w))
    error_grad = 0
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = x ** i
    return Xd

# Main script

# Problem 1
# load the sample data
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
lambdas = np.linspace(0, 0.004, num=k)
k = 21
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)    
    i = i + 1
plt.plot(lambdas,rmses4)

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
