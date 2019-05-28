LeNet-5网络

1.定义图像输入函数，即输入data_batch_1的10000个32*32*3图像
labels：0-9表示10个类别
filenames：10000个图像名称
batch_label: batch序号

2.构造网络所需的函数，包括：
权重w初始化，tf.random_normal为根据shape产生随机分布,均值标准差自定, shape=[height, width, channels, filters]；
参数b初始化, tf.constant为根据shape产生常量；
卷积函数, x=[m, height, width, channels], m为待训练集数量；
最大池化函数；
计算准确率, 根据测试集

3.利用tf占位符定义输入xs与ys

4.构造网络：
从输入到卷积、池化、再卷积、再池化、最后拉伸，然后放入两层全卷积再输出

5.计算预测函数与代价函数

6.每步训练后，梯度会下降，使代价函数达到最小值

7.训练data_batch_1的10000个32*32*3图像，每32步为一次迭代，每10次迭代显示迭代次数、代价和精确度，最终10000张图片训练完后的精确度大约为10%

