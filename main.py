import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.2, X[0, :].max() + 0.2
    y_min, y_max = X[1, :].min() - 0.2, X[1, :].max() + 0.2
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


# https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb

def sig(x):
 return 1/(1 + np.exp(-x))

def sig_d(x):
 return sig(x)*(1-sig(x))

def relu(x):
    if x >= 0:
        return x
    else:
        return 0
relu_array = np.vectorize(relu)

def relu_d(x):
    if x >= 0:
        return 1
    else:
        return 0
relu_d_array = np.vectorize(relu_d)

# compute the model on an array of input
def model(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu_array(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sig(Z2)

    return A2

# get info of dot identity from a picture values
def Y_from_img(img, X):
    m = X.shape[1]
    max = img.max()

    Y = np.zeros((1,m))
    for i in range(m):
        prob = img[X[0,i], X[1,i]] / max
        Y[0,i] = random.random() < prob

    return Y

# print points on top of picture
def show_points(img, X, Y):
    n = X.shape[1]
    fig, ax = plt.subplots()

    ax.imshow(img, cmap='binary')
    ax.scatter(X[1,], X[0,], s=1.5, c=Y[0,], cmap='bwr')

    #for i in range(n):
    #    ax.annotate(str(Y[0,i]), (X[0,i], X[1,i]))

    plt.show()


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    # n_x neurons as input
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))

    # n_h neurons in hidden layer

    # n_y neurons of output
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sig(Z2)

    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}

    return(A2, cache)

def compute_cost(A2, Y, parameters):
    # number of examples
    m = Y.shape[1]

    W1 = parameters['W1']
    w2 = parameters['W2']

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print('cost after iteration %i: %f' % (i, cost))
            # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2

    return predictions

if __name__ == '__main__':

    img = cv2.imread('area2.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = 1000
    X = np.random.randint(0, 512, (2,m))
    # X = np.array(np.meshgrid(np.array(range(0,100)), np.array(range(0,100)))).reshape(2,10000) * 5
    Y = Y_from_img(img, X)

    show_points(img, X, Y)

    X = (X - 256) / 128.0

    p = nn_model(X, Y, n_h =20, num_iterations = 100000, print_cost=True)
    plot_decision_boundary(lambda x: predict(p, x.T), X, Y)
    show_points(img, X*128 + 256, predict(p, X))



