# coding:utf-8
# @date: 19-5-25

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 只显示警告信息和错误信息，warning与error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 创建x占位符，用于临时存放MNIST图片的数据，
# [None, 784]中的None表示不限长度，而784则是一张图片的大小（28×28=784）
x = tf.placeholder(tf.float32, [None, 784])
# W存放的是模型的参数，也就是权重，一张图片有784个像素作为输入数据，而输出为10
# 因为（0～9）有10个结果
# b则存放偏置项,y=x*W+b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y表示softmax回归模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_存的是实际图像的标签，即对应于每张输入图片实际的值
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义损失函数，这里用交叉熵来做损失函数，y存的是我们训练的结果，而y_存的是实际标签的值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 优化函数，这里我们使用梯度下降法进行优化，0.01表示梯度下降优化器的学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 将训练结果保存，如果不保存我们这次训练结束后的结果也随着程序运行结束而释放了
saver = tf.train.Saver()

# 上面所做的只是定义算法，并没有真的运行，tensorflow的运行都是在会话（Session）中进行
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 开始训练，这里训练一千次
    for _ in range(1000):
        # 每次取100张图片数据和对应的标签用于训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 将取到的数据进行训练
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(W))
    print(sess.run(b))

    #saver.restore(sess, './saver/mnist.ckpt')
    # 检测训练结果，tf.argmax取出数组中最大值的下标，tf.equal再对比下标是否一样即可知道预测是否正确
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # correct_prediction得到的结果是True或者False的数组
    # 我们再经过tf.cast将其转为数字的形式，即将[True, True, Flase, Flase]转成[1, 1, 0, 0]
    # 最后用tf.reduce_mean计算数组中所有元素的平均值，即预测的准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 开始预测运算，并将准确率输出
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # 最后，将会话保存下来
    saver.save(sess, './saver/mnist.ckpt')

    # path = saver.save(
    #     sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
    #     write_meta_graph=False, write_state=False)
    # print('Saved:', path)
