'''CREDITS
1) Thanks to Andrew Ng's Machine Learning Course for teaching me the content
2) Thanks to Dibgerge on github.com for insanely helpful python versions of Andrew Ng's assignments.
A lot of the stuff pertaining to multivariate linear regression comes pretty much directly from him as this
is what I'm used to. Link: https://github.com/dibgerge/ml-coursera-python-assignments

'''

# used for manipulating directory paths
import os

# scientific and vector computation for python
import numpy as np  

# plotting lib (note: people usually use import matplotlib.pyplot as plt)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D # needed to plot 3D surfaces

data = 0

def ask_data(data):
    '''Asks users for activities done in a day. For now it stores in a csv 
    
    temp restriction: have to say 2 features, the number is number of hours the user did the activity
    also for now, the only activities I do in a day are code and relax
    returns: None
    '''
    if data is 0:
        data = np.zeros(3)
        data[0] = input("Hours spent on coding: ")
        data[1] = input("Hours spent on relaxing: ")
        data[2] = input("Quality of day: ")
        return data
    
    temp = np.zeros(3)
    temp[0] = input("Hours spent on coding: ")
    temp[1] = input("Hours spent on relaxing: ")
    temp[2] = input("Quality of day: ")
    
    data = np.vstack([data, temp])
    return data

'''
for i in range(4):
    data = ask_data(data)
'''
data = np.array([[3. , 3. , 6.7],
       [2. , 4. , 7. ],
       [2. , 3. , 8. ],
       [1. , 5. , 7. ]])

X, y = data[:, :2], data[:, 2]
m = y.size    

def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    # we need the mean and std deriv in future in case another example (m) gets added
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    for i in range(0, X_norm.shape[1]):
        # compute mean
        mu[i] = np.mean(X_norm[:, i])
        # subract mean
        X_norm[:, i] = X_norm[:, i] - mu[i]
        # compute standard derivation
        sigma[i] = np.std(X_norm[:, i])
        # divide by stadard derivation
        X_norm[:, i] = (X_norm[:, i])/sigma[i]
    # ================================================================
    return X_norm, mu, sigma

# call featureNormalize on the loaded data
'''
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)
'''
# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    hyp = X.dot(theta)
    # this is vectorized version of fomula that works w multiple variables
    J = 1/(2*m)*((hyp-y).T).dot((hyp-y))
    
    # ==================================================================
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
  
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        hyp = X.dot(theta)
        # vectorized formula
        theta = theta - alpha*(1/m)*(X.T.dot(hyp-y))
        
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history

