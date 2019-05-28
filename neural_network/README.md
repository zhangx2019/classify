神经网络算法
1.预测函数，predict function，就是你设计的模型函数，如predict function(w, b) = w*x + b，w,b为矩阵: 

 a1 = activation(w1*x + b1) 
 
 a2 = activation(w2 * a1+ b2) 
 
 predict function = a2 
 
 activation为激活函数，一般为使目标非线性化，如sigmoid(x) = 1/(1+e^-x)，ReLU(x) = max(0, x)

2.计算合理的参数，如w1,b1,w2,b2等，常用的是梯度下降法，神经网络Back Propagation算法: 

w1 := w1 - rate*dw1 

w2 := w2 - rate*dw2 

b1 := b1 - rate*db1 

b2 := b2 - rate*db2 

理论上只要能合理的计算参数，使预测函数朝着最优前进的参数即可

3.计算代价函数cost function，使之尽量小，如 
cost function = (1/m) * 累加符（predict function(i) - y(i)）^2, i = 1,2,3,…,m.(m为训练集个数）

Back Propagation算法
给定训练集{(x1, y1), (x2, y2), (x3, y3), … ,(xm, ym)}, 以及你自己搭建的神经网络, 共L层
1.前向传播，计算出每层单元al, 这里的x, w, b, a, z都是矩阵，xi为训练集(xi, yi), 如： 
a0 = xi；
z1 = w1 * x + b1; 

a1 = activation1(z1); 

z2 = w2 * a1 + b2; 

a2 = activation2(z2); 
… 

zl = wl * al-1 + bl; 

al = activationl(zl); 
…. 

zL = wL * aL-1 + bL; 

aLi = activationL(zL)

2.计算每一层单元的误差delta, yi为训练集(xi, yi), activationT(x)为activation(x)的导函数，如: 

deltaL = aLi - yi; 

deltaL-1 = (wL.T * deltaL) .* activationL(aL-1); 

deltaL-2 = (wL-1.T * deltaL-1) .* activationL-1(aL-2); 
… 

delta1 = (w2.T * delta2) .* activation2(a1)

3.将每一层单元的误差传递给参数w,b, D初始化为0, 如上:
 
D1 = D1 + delta1 * a0.T = D1 + delta1 * x.T;
 
D2 = D2 + delta2 * a1.T; 

D3 = D3 + delta3 * a2.T; 
… 

DL = DL + deltaL-1 * aL-1.T;

4.将1,2,3遍历m次后，更新参数w,b, 如上: 

w1 = w1 - rate * (1/m) * (D1 + lambda * w1); 

b1 = b1 - rate * delta1; 

w2 = w2 - rate * (1/m) * (D2 + lambda * w2); 

b2 = b2 - rate * delta2; 
… 

wL = wL - rate * (1/m) * (DL + lambda * wL); 

bL = bL - rate * deltaL;







