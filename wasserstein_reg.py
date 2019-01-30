'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

pi = 3.141592

import random

def loss(Y, T):
    return torch.sum((Y - T) ** 2)


def linear(x,  w_0, w_1):
    return w_0 + torch.mul(w_1 , x)


w_t =  torch.Tensor([10, 4.4])

count = 0


X_tmp = np.arange(-10.0, 20.0, 0.01)

X = torch.Tensor(X_tmp)
T = linear (X, w_t[0], w_t[1])
#plt.plot(X.numpy(), T.numpy())
#plt.show()


batch_size = 20
step_count = 1000

w_0 = torch.tensor(5.0)
w_1 = torch.tensor(2.0)

result_param =[]

alpha = 0.001

for i in range(0,step_count):

    offset = random.randint(1,len(X)-batch_size)

    w_0_t = torch.tensor(w_0.clone().detach(), requires_grad=True, dtype=torch.float64)
    w_1_t =  torch.tensor(w_1.clone().detach(), requires_grad=True, dtype=torch.float64)
    #print(w_mu_t)
    #print(w_sig_t)

    Y = w_0_t + torch.mul(X[offset], w_1_t)
    result = loss(Y,T[offset])
    #print(result)
    result.backward()
    if(i % 10 == 0):
        result_param.append(result)
        print("-------------------------")
        print(i)
        print(result)
        print(w_0)
        print(w_1)
    #SGD 
    #w_0 = w_0 - alpha * w_0_t.grad
    #w_1 = w_1 - alpha * w_1_t.grad

    #Wasserstein
    was_mat_00 = np.sqrt(pi) / (w_1 * w_1)
    was_mat_11 = (T[offset]*T[offset] + np.sqrt(pi) -2* w_0 * T[offset] * np.sqrt(pi) + w_0 * w_0 ) / np.power(w_1, 5.)

    was_mat_01 = -2*(T[offset] - w_0 * np.sqrt(pi)) / np.power(w_1, 4.)
    w_0 = w_0 - alpha* ( was_mat_00 * w_0_t.grad + was_mat_01 * w_1_t.grad)
    w_1 = w_1 - alpha*(  was_mat_01 * w_0_t.grad+ was_mat_11* w_1_t.grad)
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
