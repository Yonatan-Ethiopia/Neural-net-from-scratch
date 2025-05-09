import numpy as np

inputs = np.array([[0,1], [1,0], [1,1], [0,0]])

target =np.array([ 1, 1, 0, 0])

weights = np.random.randn(2,1)
bias = 1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def predict(x):
    z = np.dot(x, weights)+bias
    prediction = sigmoid(z)
    return prediction 
