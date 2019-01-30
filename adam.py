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

X_tmp = np.arange(-10.0, 20.0, 0.01)

X = torch.Tensor(X_tmp)
T = gaussian (X, w_t[0], w_t[1])


T = gaussian (X, w_t[0], w_t[1])



batch_size = 20
step_count = 7000


w_mu = -3.0
w_sig = 2.0

result_param =[]

alpha = 0.1


w_mu_t = torch.tensor(w_mu, requires_grad=True)
w_sig_t =  torch.tensor(w_sig, requires_grad=True)


for i in range(0,step_count):

    offset = random.randint(1,90)

    Y = 1./(np.sqrt(2.*pi)*w_sig_t)*torch.exp(-torch.pow((X - w_mu_t)/w_sig_t, 2.)/2)


    result = loss(Y,T)
    result.backward()

    print(result)

    if(i % 10 == 0):
        result_param.append(result)
    #print(w_mu_t)
    #print(w_sig_t)

    optimizer = torch.optim.Adam([w_mu_t,w_sig_t], lr=0.1)
    optimizer.step()



def gaussian_np(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

t = np.arange(-20, 40, 0.1)
'''
plt.plot(t, gaussian_np(t,3,6))
plt.plot(t, gaussian_np(t,30, 2))
plt.show()
'''

step = range(0, len(result_param)*10,10)

plt.plot(step, result_param)
plt.show()
