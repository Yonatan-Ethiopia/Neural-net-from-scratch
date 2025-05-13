import numpy as np

inputs = np.array([[0,1], [1,0], [1,1], [0,0]])

target =np.array([ 1, 1, 0, 0]).reshape(-1,1)
epoch = 1000
weights = np.random.randn(2,1)
bias = 1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def predict(x):
    z = np.dot(x, weights)+bias
    prediction = sigmoid(z)
    return prediction, z
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip( y_pred, epsilon, 1- epsilon )
    loss = y_true*-1*np.log(y_pred)+(1-y_true)*-1*np.log(1-y_pred)
p, z = predict(inputs)
def backprop(x, w, b, p, t, z):
    dLdW = ( -t/p +(1-t)/(1-p))*(p*(1-p))*x
    dLDb = ( -t/p +(1-t)/(1-p))*(p*(1-p))*b
    return dLdW, dLDb

for i in epoch:
    p, z = predict(inputs)
    loss = binary_cross_entropy( target, p)
    print(i)
