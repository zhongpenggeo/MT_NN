'''
neural network
'''

import numpy as np

eta = np.finfo(np.float32).eps

# normaliation
def input_max(X,lb,ub):
    X = X/ub
    return X
def input_min(X,lb,ub):
    X = (X-lb)/(ub-lb)
    return X
def input_minmax(X,lb,ub):
    X = 2.0* (X-lb)/(ub-lb) - 1.0
    return X
def input_log(X,lb,ub):
    return torch.log10(X)/np.log10(ub)

# build a network
class FNN(object):
    def __init__(self,net_layers,activation_func,net_type="complex"):
        w_0 = np.random.uniform(-1,1,size=(net_layers[1],net_layers[0]))
        b_0 = np.random.uniform(-1,1,size=(net_layers[1],1))
        II = np.array([0+1j],dtype=complex)
        if net_type == "real":
            w_1 = np.zeros((net_layers[1],net_layers[0]))
            b_1 = np.zeros((net_layers[1],1))
        elif net_type == "complex":
            phase_w =np.random.uniform(-np.pi,np.pi,size=(net_layers[1],net_layers[0]))
            phase_b =np.random.uniform(-np.pi,np.pi,size=(net_layers[1],1))
            w_0 = w_0 * np.cos(phase_w)
            w_1 = w_0 * np.sin(phase_w)
            b_0 = b_0 * np.cos(phase_b)
            b_1 = b_0 * np.sin(phase_b)
        else:
            raise KeyError('bad net types, just real or complex') 
        self.w0 = w_0+w_1*II
        self.b0 = b_0+b_1*II
        w_2 = np.random.uniform(size=(net_layers[2],net_layers[1]))
        w_3 = np.random.uniform(size=(net_layers[2],net_layers[1]))
        self.w1 = w_2+w_3*II
    def __call__(self,x,lb,ub):
        x = input_minmax(x,lb,ub)
        X1 = np.tanh(np.dot(x,self.w0.T)+self.b0.T)
        Y  = np.dot(X1,self.w1.T)
        # X1: the output of hidden layer
        # Y: the output of last(output) layer
        return X1,Y