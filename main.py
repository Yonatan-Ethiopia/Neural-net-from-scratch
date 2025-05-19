import numpy as np

inputs = np.array([[0,1], [1,0], [1,1], [0,0]])
lnr_initial = 0.1
decay_rate = 0.001
target =np.array([ 1, 1, 0, 0]).reshape(-1,1)
epoch = 1000
w1= np.random.randn(2,1)
w2 = np.random.randn(2,1)
w3 = np.random.randn(2,1)
b1 = 1
b2 = 0.75
b3 = 0.5
def sigmoid(x):
    return 1/(1+np.exp(-x))
def A1(x):
    z = np.dot(x, w1)+b1
    a1= sigmoid(z)
    return a1, z
def A2(x):
    z = np.dot( x, w2.T)+b2
    a2 = sigmoid(z)
    return a2, z
def A3(x):
    z = np.dot( x, w3)+b3
    prediction= sigmoid(z)
    return prediction, z
a1, z1 = A1(inputs)
a2, z2 = A2(a1)
a3, z3 = A3(a2)
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip( y_pred, epsilon, 1- epsilon )
    loss = y_true*-1*np.log(y_pred)+(1-y_true)*-1*np.log(1-y_pred)

def backprop(x, p, t):
    m = len(x)
    dZ = p - t
    dz2 = sigmoid_derivative(z2)
    dz3 = sigmoid_derivative(z3)
    dW = (1/m) * np.dot(x.T, dZ)
    dw2 = np.dot( A2.T, dz2)
    dw3 = np.dot( A3.T, dz3)
    dB = (1/m) * np.sum(dZ)
    return dW,dB
for i in range(1, 1+epoch):
    lnr = lnr_initial/(1 + decay_rate*i)
    p, z = A3(a2)
    loss = binary_cross_entropy( target, p)
    dw,db = backprop(inputs, p, target)
    w1 -= lnr*dw
    b1 -= lnr*db
    if( epoch%10 == 0):
        print("weight : ",w1)
        print("bias : ",b1)
        print("learningrate : ",lnr)
        print("   Prediction:  ", p)
