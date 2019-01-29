import numpy as np
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
 
from scipy.stats import norm
import matplotlib.pyplot as plt

pi = 3.14159265

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)



def dev_mu_gaussian(x, mu, sig):
    return -(x-mu)/sig * gaussian(x,mu,sig)


def dev_sig_gaussian(x, mu, sig):
    return (-1/(2*sig) + (x-mu)/np.power(sig, 2.)  * gaussian(x,mu,sig)

# Data comes from y = f(x) = [2, 3].x + [5, 7]
X = np.random.randn(100, 1)
w =  np.array([3,6])
w_t =  np.array([-5, 2])


T = gaussian(X, w_t[0], w_t[1])


X, T = shuffle(X, T)

X_train, X_test = X[:150], X[:50]
T_train, T_test = T[:150], T:50]

print(X_train)
print(T_train)


alpha = 0.01

# Training
for it in range(20):
    # Forward

    Y = gaussian(X, w[0], w[1])

    loss = NLL(y, t_train)

    # Loss
    print(f'Loss: {loss:.3f}')

    m = y.shape[0]

    dy = (y-t_train)/(m * (y - y*y))
    dz = sigm(z)*(1-sigm(z))
    dW = X_train.T @ (dz * dy)

    grad_loglik_z = (t_train-y)/(y - y*y) * dz
    grad_loglik_W = grad_loglik_z * X_train
    F = np.cov(grad_loglik_W.T)

    # Step
    W = W - alpha * np.linalg.inv(F) @ dW
    # W = W - alpha * dW

# print(W)

y = sigm(X_test @ W).ravel()
acc = np.mean((y >= 0.5) == t_test.ravel())

print(f'Accuracy: {acc:.3f}')



'''
def sigm(x):
    return 1/(1+np.exp(-x))


def NLL(y, t):
    return -np.mean(t*np.log(y) + (1-t)*np.log(1-y))


alpha = 0.01

# Training
for it in range(20):
    # Forward
    z = X_train @ W
    y = sigm(z)
    print(y)
    loss = NLL(y, t_train)

    # Loss
    print(f'Loss: {loss:.3f}')

    m = y.shape[0]

    dy = (y-t_train)/(m * (y - y*y))
    dz = sigm(z)*(1-sigm(z))
    dW = X_train.T @ (dz * dy)

    grad_loglik_z = (t_train-y)/(y - y*y) * dz
    grad_loglik_W = grad_loglik_z * X_train
    F = np.cov(grad_loglik_W.T)

    # Step
    W = W - alpha * np.linalg.inv(F) @ dW
    # W = W - alpha * dW

# print(W)

y = sigm(X_test @ W).ravel()
acc = np.mean((y >= 0.5) == t_test.ravel())

print(f'Accuracy: {acc:.3f}')
'''