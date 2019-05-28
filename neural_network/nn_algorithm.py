#三层网络，个数输入层1，隐藏层3，5，输出层1
import numpy as np
import matplotlib.pyplot as plt

#前向传播网络构造
def add_layer(previous_layer_units,weight,bias,activation_function=None):
    Wx_b_plus=np.dot(weight,previous_layer_units)+bias
    if activation_function is None:
        next_layer_units=Wx_b_plus
    else:
        next_layer_units=activation_function(Wx_b_plus)
    return next_layer_units

#激活函数sigmoid
def Sigmoid(x):
    return 1/(1+np.exp(-x))
#激活函数的导数，x为矩阵点乘
def SigmoidT(x):
    return x*(1-x)

#反向传播，back propagation
def back_propagation(x,y,m,w1,b1,w2,b2,w3,b3,lambd,rate):
    D1=np.zeros(w1.shape)
    D2=np.zeros(w2.shape)
    D3=np.zeros(w3.shape)
    for i in range(m):  #m个训练集{(x1, y1), (x2, y2), (x3, y3), … ,(xm, ym)}
        #1.前向传播, forward propagation
        a1=add_layer(x[i],w1,b1,activation_function=Sigmoid)
        a2 = add_layer(a1, w2, b2, activation_function=Sigmoid)
        a3 = add_layer(a2, w3, b3, activation_function=Sigmoid)

        # 2.计算每一层每个单元的误差delta, computing delta
        delta3 = a3 - y[i]
        delta2 = np.dot(w3.T, delta3) * SigmoidT(a2)
        delta1 = np.dot(w2.T, delta2) * SigmoidT(a1)

        # 3.利用每一层单元的误差, 将其传递给权重weight, 并累加
        D1 = D1 + np.dot(delta1, x[i].T)
        D2 = D2 + np.dot(delta2, a1.T)
        D3 = D3 + np.dot(delta3, a2.T)

    # 4.计算平均值, 加入超参数lambd控制更新，dw = (1/m) * (D + lambda * w)
    D1 = (1 / m) * (D1 + lambd * w1)
    D2 = (1 / m) * (D2 + lambd * w2)
    D3 = (1 / m) * (D3 + lambd * w3)

    #5.更新权重, 加入梯度下降速率rate
    w1 = w1 - rate * D1
    b1 = b1 - rate * delta1
    w2 = w2 - rate * D2
    b2 = b2 - rate * delta2
    w3 = w3 - rate * D3
    b3 = b3 - rate * delta3

    #6.计算成本函数, compute cost function
    J = 0
    for i in range(m):
        a1 = add_layer(x[i], w1, b1, activation_function=Sigmoid)
        a2 = add_layer(a1, w2, b2, activation_function=Sigmoid)
        a3 = add_layer(a2, w3, b3, activation_function=Sigmoid)
        J = J + (a3 - y[i]) * (a3 - y[i])
    J = J / m
    return J, w1, b1, w2, b2, w3, b3

#训练
def train(x, y, m, lambd, rate, iters, ax):
    w1 = np.random.rand(3, 1) * 0.01
    b1 = np.zeros((3, 1))
    w2 = np.random.rand(5, 3) * 0.01
    b2 = np.zeros((5, 1))
    w3 = np.random.rand(1, 5) * 0.01
    b3 = np.zeros((1, 1))
    for i in range(iters):
        J, w1, b1, w2, b2, w3, b3 = back_propagation(x, y, m, w1, b1, w2, b2, w3, b3, lambd, rate)
        print(J) #打印loss
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = np.zeros(x.shape)
        for i in range(m):
            a1 = add_layer(x[i], w1, b1, activation_function=Sigmoid)
            a2 = add_layer(a1, w2, b2, activation_function=Sigmoid)
            a3 = add_layer(a2, w3, b3, activation_function=Sigmoid)
            prediction_value[i] = a3
        lines = ax.plot(x, prediction_value, 'r-', lw=5)
        plt.pause(0.2)


x_data = np.linspace(-10, 10, 200, dtype=np.float32)
y = np.zeros(x_data.shape)
for i in range(200):
    if x_data[i] < 0:
        y[i] = 1
    else:
        y[i] = 0

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y)
plt.show()

train(x_data, y, 200, 0, 0.5, 100, ax)






