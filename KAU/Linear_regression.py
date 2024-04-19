import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

eta = 0.1 # hyperparameter
iter_num = 100000
Ddim = 3
Ndim = 100

def theta_gen(Ddim):
    return np.random.rand(Ddim+1,1)*100

def X_standard_gen(X): # 표준화 전처리
    return (X-np.mean(X,axis=0))/(np.std(X,axis=0))

def MSE_gradient_V1(X_b,yr,theta):
    m = Ndim
    yr = yr[:,np.newaxis]
    gradient = 2/m * X_b.T @ ((X_b@theta) - yr)
    return gradient

def get_batch_size(random_shuffle, size):
    np.random.shuffle(random_shuffle)
    Xr_b = random_shuffle[:,0:Ddim+1]
    yr = random_shuffle[:,-1]
    splitX = np.array_split(Xr_b, size, axis=0)
    splity = np.array_split(yr, size, axis=0)
    return splitX, splity

def X_concatenate(X):
    M,N = X.shape
    one = np.ones((M,1))
    X_b = np.c_[X,one]
    X_bT = X_b.T
    return X_b

# 초기 셋팅 (다항 회귀분석으로 접근하도록)
def random_function(Ddim,Ndim):
    x_ndim = np.arange(1,Ndim+1)
    X = np.zeros((Ndim, Ddim))

    for i in range(Ddim):
        for j in range(Ndim):
            X[j,i] = x_ndim[j] ** ((i+1)/3)

    np.random.seed(25)
    noise = np.random.rand(Ndim,1) * 5
    weight = np.random.rand(Ddim,1)

    y = X@weight + noise
    return X ,y


X, y = random_function(Ddim,Ndim)

plt.plot(X[:,0],y,'ro',label='$x_0$')
plt.plot(X[:,1],y,'go',label='$x_1$')
plt.plot(X[:,2],y,'bo',label='$x_2$')
plt.legend()

theta = theta_gen(Ddim)
print(theta)

X_standardization = X_standard_gen(X)
standard_c = np.c_[X_standardization,y]

train_size = 60
np.random.shuffle(standard_c)
train_data, test_data = standard_c[0:train_size,:], standard_c[train_size:Ndim,:]

X_train, y_train = train_data[:,0:Ddim], train_data[:,-1]
X_test, y_test = test_data[:,0:Ddim], test_data[:,-1]

# LMSE
X_b = X_concatenate(X)

best_theta = inv(X_b.T @ X_b) @ (X_b.T @ y)
print(best_theta) # best_theta[0] ~ [2]: 가중치 w, [3]: bias

MLS_predict = X @ best_theta[0:Ddim] + best_theta[-1]


plt.plot(y, MLS_predict, 'ro',markersize=3)
plt.plot([5,35],[5,35], 'b--')
plt.xlabel('y')
plt.ylabel('MLS_predicted')
plt.show()

# GDC
def gradient_decent(theta, X, y):
    X_b = X_concatenate(X)
    random_shuffle = np.c_[X_b, y]
    x_size = X.shape[0]
    gradient_decent_history = []
    theta_d = np.copy(theta)

    for m in range(iter_num):
        X_split, y_split = get_batch_size(random_shuffle,x_size)
        gradient_decent_history.append(np.copy(theta_d))
        for shuffle_data in zip(X_split,y_split):
            one_data, one_result = shuffle_data
            grad = MSE_gradient_V1(one_data,one_result,theta_d)
            theta_d = theta_d - eta * grad

    return theta_d, np.array(gradient_decent_history)

theta_d, gradient_decent_history= gradient_decent(theta,X_train,y_train)

# RMSProp
# 변수끼리 충돌되는 상황이 발생

def RMSProp_gen(theta_RMS, X, y):
    theta = np.copy(theta_RMS)
    decay_rate = 0.9
    h = 0
    x_size = X.shape[0]
    X_b = X_concatenate(X)
    random_shuffle = np.c_[X_b, y]
    RMS_history = []
    for i in range(iter_num):
        X_split, y_split = get_batch_size(random_shuffle,x_size)
        RMS_history.append(np.copy(theta))
        for shuffle_data in zip(X_split,y_split):
            one_data, one_result = shuffle_data
            grad = MSE_gradient_V1(one_data,one_result,theta)
            h = decay_rate*h + (1-decay_rate)*(grad**2) # moving average 개념을 추가함
            theta = theta -  eta * grad / (np.sqrt(h)+1e-7)
    return theta, np.array(RMS_history)

RMSProp, RMS_history = RMSProp_gen(theta,X_train,y_train)

# Adam
def Adam_gen(theta_Adam, X, y):
    theta = np.copy(theta_Adam)
    decay_rate1 = 0.9
    decay_rate2 = 0.9
    s, r = 0, 0
    x_size = X.shape[0]
    X_b = X_concatenate(X)
    random_shuffle = np.c_[X_b, y]
    Adam_history = []
    for t in range(1,iter_num+1):
        X_split, y_split = get_batch_size(random_shuffle,x_size)
        Adam_history.append(np.copy(theta))
        for shuffle_data in zip(X_split,y_split):
            one_data, one_result = shuffle_data
            grad = MSE_gradient_V1(one_data,one_result,theta)
            s = decay_rate1 * s + (1-decay_rate2) * grad
            r = decay_rate2 * r + (1-decay_rate2)*(grad**2) # moving average 개념을 추가함
            Correct_bias1 = s / (1-decay_rate1**t)
            Correct_bias2 = r / (1-decay_rate2**t)
            theta = theta -  eta * Correct_bias1 / (np.sqrt(Correct_bias2)+1e-7)
    return theta, np.array(Adam_history)

Adam, Adam_history = Adam_gen(theta,X_train,y_train)

# AdaGrad
def AdaGrad_gen(theta_Ada, X,y):
    theta = np.copy(theta_Ada)
    x_size = X.shape[0]
    h = 0
    X_b = X_concatenate(X)
    random_shuffle = np.c_[X_b, y]
    Ada_history = []
    for i in range(iter_num):
        X_split, y_split = get_batch_size(random_shuffle,x_size)
        Ada_history.append(np.copy(theta))
        for shuffle_data in zip(X_split,y_split):
            one_data, one_result = shuffle_data
            grad = MSE_gradient_V1(one_data,one_result,theta)
            h += (grad*grad)
            theta = theta -  eta * grad / (np.sqrt(h+1e-7))
    return theta, np.array(Ada_history)

AdaGrad, Ada_history = AdaGrad_gen(theta,X_train,y_train)
