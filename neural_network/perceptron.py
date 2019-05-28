import numpy as np
import matplotlib.pyplot as plt

n=0  #迭代次数
lr=0.10  #学习速率
#输入数据
X=np.array([[1,1,2,3],
            [1,1,4,5],
            [1,1,1,1],
            [1,1,5,3],
            [1,1,0,1]])
#标签
Y=np.array([1,1,-1,1,-1])
print(Y.T)
#print(np.random.random(X.shape[1])) #0-1
# print(np.random.rand(X.shape[0]))   #0-1
# print(np.random.randn(X.shape[0]))  #-1-1
#权重初始化，取值范围为-1~1
W=(np.random.random(X.shape[1])-0.5)*2
# print(W.T)
# print(np.dot(X, W.T))
# print(np.sign(np.dot(X,W.T)))
# output=np.sign(np.dot(X,W.T))
# print(output.T)
# w1=(Y-output.T).dot(X)
# print(w1)
# print(int(X.shape[0]))
# all_x=X[:, 2]
# all_y=X[:, 3]
# all_negative_x=[1,0]
# all_negative_y=[1,1]
# print(all_x,all_y,all_negative_x,all_negative_y)

def get_show():
    #正样本
    all_x=X[:, 2]
    all_y=X[:, 3]

    #负样本
    all_negative_x=[1,0]
    all_negative_y=[1,1]

    #计算分界线斜率与截距
    k=-W[2]/W[3]
    b=-(W[0]+W[1])/W[3]
    #生成x刻度
    xdata=np.linspace(0,5)
    plt.figure()
    plt.plot(xdata,xdata*k+b,'r')
    plt.plot(all_x,all_y,'bo')
    plt.plot(all_negative_x,all_negative_y,'yo')
    plt.show()


# 更新权值函数
def get_update():
    # 定义所有全局变量
    global X, Y, W, lr, n
    n += 1
    # 计算符号函数输出
    new_output = np.sign(np.dot(X, W.T))
    # 更新权重
    new_W = W + lr * ((Y - new_output.T).dot(X)) / int(X.shape[0])
    W = new_W


def main():
    for _ in range(100):
        get_update()
        new_output = np.sign(np.dot(X, W.T))
        if (new_output == Y.T).all():
            print('迭代次数：', n)
            break
    get_show()


if __name__ == '__main__':
    main()
