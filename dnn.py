import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

#blurred = gaussian_filter(a, sigma=7)

pi = 3.141592

import random

def loss(Y, T):
    return torch.sum((Y - T) ** 2)


def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*pi)*sig)*torch.exp(-torch.pow((x - mu)/sig, 2.)/2)

w_t =  torch.Tensor([10, 4.4])

count = 0


X_tmp = np.arange(-10.0, 20.0, 0.01)

X = torch.Tensor(X_tmp)
T = gaussian (X, w_t[0], w_t[1])

#plt.plot(X.numpy(), T.numpy())
#plt.show()


batch_size = 20
step_count = 10000

w_mu = torch.tensor(-3.0)
w_sig = torch.tensor(2.0)

result_param =[]

alpha = 1.0

for i in range(0,step_count):

    offset = random.randint(1,len(X)-batch_size)

    w_mu_t = torch.tensor(w_mu.clone().detach(), requires_grad=True)
    w_sig_t =  torch.tensor(w_sig.clone().detach(), requires_grad=True)
    #print(w_mu_t)
    #print(w_sig_t)
    Y = 1./(np.sqrt(2.*pi)*w_sig_t)*torch.exp(-torch.pow((X[offset:offset+batch_size] - w_mu_t)/w_sig_t, 2.)/2)
    #print(Y)


    result = loss(Y,T[offset:offset+batch_size])
    result.backward()
    if(i % 10 == 0):
        result_param.append(result)
        print(result)
        print(i)
        print(w_mu)
        print(w_sig)
    #SGD 
    #w_mu =  w_mu - alpha * w_mu_t.grad
    #w_sig = w_sig - alpha * w_sig_t.grad
    #print(w_mu_t.grad)

    #Wasserstein
    w_mu = w_mu - alpha* w_mu_t.grad
    w_sig = w_sig - alpha*1/(4*w_sig_t)* w_sig_t.grad
    #KL
    #w_mu = w_mu - alpha*1/(w_sig_t*w_sig_t)* w_mu_t.grad
    #w_sig = w_sig - alpha*2/(w_sig_t*w_sig_t)* w_sig_t.grad



def gaussian_np(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

t = np.arange(-20, 40, 0.1)


step = range(0, len(result_param)*10,10)

plt.plot(step, result_param)
plt.show()



'''
plt.plot(t, gaussian_np(t,3,6))
plt.plot(t, gaussian_np(t,30, 2))
plt.show()
'''
