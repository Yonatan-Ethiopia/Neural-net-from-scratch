import numpy as np

inputs = np.array([[0,1], [1,0], [1,1], [0,0]])
lnr_initial = 0.1
decay_rate = 0.001
target =np.array([ 1, 1, 0, 0]).reshape(-1,1)
epoch = 500
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
def backprop(x, p, t):
    m = len(x)
    dZ = p - t
    dW = (1/m) * np.dot(x.T, dZ)
    dB = (1/m) * np.sum(dZ)
    return dW,dB

for i in range(1, 1+epoch):
    lnr = lnr_initial/(1 + decay_rate*i)
    p, z = predict(inputs)
    loss = binary_cross_entropy( target, p)
    dw,db = backprop(inputs, p, target)
    weights -= lnr*dw
    bias -= lnr*db
    if( epoch%10 == 0):
        print("weight : ",weights)
        print("bias : ",bias)
        print("learningrate : ",lnr)
