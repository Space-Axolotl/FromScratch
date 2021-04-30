import numpy as np
import matplotlib.pyplot as plt

def weightInitialization(n_features):
    """
    initialize all the necessery variables
    """
    w = np.zeros(n_features)
    b = .0
    return w,b

def sigmoid(z):
    """
    sigmoid function
    """
    return 1/(1+np.exp(-z))

def cost_func(w, b, X, Y):
    """
    
    """
    # Number of elements in X
    m = X.shape[0]
    
    # Prediction
    Y_hat = sigmoid(w*X.T+b)

    # Cost function
    cost = (-1/m)*(np.sum((Y.T*np.log(Y_hat)) + ((1-Y.T)*(np.log(1-Y_hat)))))

    return cost

def grad_descent(w,b,X,Y,learning_rate):
    """Gradient Descent
    w - list of weights
    b - float bias
    X - training features
    Y - training labels
    learning rate - gradient step size

    returns list of weights and float bias
    """
    # Number of elements in X
    m = X.shape[0]

    # Prediction
    Y_hat = sigmoid(w*X.T+b)

    # Gradient calculation (derivatives with respect to w and b)
    dw = (1/m)*(np.dot(X.T, (Y_hat-Y.T).T))
    db = (1/m)*(np.sum(Y_hat-Y.T))
    
    # Weight update
    w = w - (learning_rate * (dw.T))
    b = b - (learning_rate * db)
    return w,b

def min_cost_func(w, b, X, Y, learning_rate, no_iterations):
    """fuction that minimizes cost calculated with cost_func

    w - list of weights
    b - float bias
    X - training features
    Y - training labels
    learning rate - gradient step size
    num of iterations - number of iterations

    returns coeff of the logistic function"""
    costs = []
    for i in range(no_iterations):
        
        cost = cost_func(w, b, X, Y)
    
        w,b = grad_descent(w,b,X,Y,learning_rate)
        
        if (i % 500 == 0):
            costs.append(cost)
            print("Cost after %i iteration is %f" %(i, cost))
    
    # Final coefficients
    coeff = [w,b]
    
    return coeff

def Logistic(X,Y,learning_rate,num_iterations):
    """Main regression module that uses, cost_func and grad_descent
    X - training features
    Y - training labels
    learning rate (best 0.1<->0.01)
    num of iterations (without error < 5600) else error: division by 0
    if num of iterations >= 5600 still works but does not output cost corectly in terminal
    
    Returns: list of coefficients of the logistic function / decision boundary"""
    w,b = weightInitialization(len(np.unique(Y))-1)
    coeff = min_cost_func(w, b, X, Y, learning_rate, num_iterations)
    return coeff

def data(lower_b, upper_b,step,percent=0.5, noise=.6):
    # Create the data
    X = np.arange(lower_b,upper_b,step)
    Y = np.zeros(len(X))

    # Turn percent p% data into 1
    p = percent
    length = int(len(Y)*p)
    Y[length:]=np.ones(len(Y)-length)

    midpoint = int(len(Y)/2)
    for _ in range(int(len(Y)*noise)):
        if Y[int(length-np.random.randint(lower_b,upper_b)*noise*6)] == 0:
            Y[int(length+np.random.randint(lower_b,upper_b)*noise*6)] =1
        else:
            Y[int(length+np.random.randint(lower_b,upper_b)*noise*6)] =0

    return X,Y

# Create data for logistic regression
X,Y = data(-6,6,0.1)

# Calculate coefficients with logistic regression
coeff = Logistic(X, Y,0.1,5000)

# take out coefficients from coeff variable
w,b = coeff

# Plot the decision boundary
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot()
ax.scatter(X,Y,c='mediumseagreen')

ax.plot(X,sigmoid(b+w*X),c='mediumblue')
plt.show()