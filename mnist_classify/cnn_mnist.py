# coding:utf-8
# @date: 19-5-25

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 只显示警告信息和错误信息，warning与error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MNIST_data_folder = "D:/coding/Image Classify/mnist_classify/mnist_data"
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)


# 初始化过滤器
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# 初始化偏置，初始化时，所有值是0.1
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷积运算，strides表示每一维度滑动的步长，一般strides[0]=strides[3]=1
# 第四个参数可选"Same"或"VALID"，“Same”表示边距使用全0填充
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 创建x占位符，用于临时存放MNIST图片的数据，
# [None, 784]中的None表示不限长度，而784则是一张图片的大小（28×28=784）
x = tf.placeholder(tf.float32, [None, 784])
# y_存的是实际图像的标签，即对应于每张输入图片实际的值
y_ = tf.placeholder(tf.float32, [None, 10])

# 将图片从784维向量重新还原为28×28的矩阵图片,
# 原因参考卷积神经网络模型图，最后一个参数代表深度，
# 因为MNIST是黑白图片，所以深度为1,
# 第一个参数为-1,表示一维的长度不限定，这样就可以灵活设置每个batch的训练的个数了
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
# 将过滤器设置成5×5×1的矩阵，
# 其中5×5表示过滤器大小，1表示深度，因为MNIST是黑白图片只有一层。所以深度为1
# 32表示我们要创建32个大小5×5×1的过滤器，经过卷积后算出32个特征图（每个过滤器得到一个特征图）
W_conv1 = weight_variable([5, 5, 1, 32])
# 有多少个特征图就有多少个偏置
b_conv1 = bias_variable([32])
# 使用conv2d函数进行卷积计算，然后再用ReLU作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 卷积以后再经过池化操作
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# 因为经过第一层卷积运算后，输出的深度为32,所以过滤器深度和下一层输出深度也做出改变
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 经过两层卷积后，图片的大小为7×7（第一层池化后输出为（28/2）×（28/2），
# 第二层池化后输出为（14/2）×（14/2））,深度为64，
# 我们在这里加入一个有1024个神经元的全连接层，所以权重W的尺寸为[7 * 7 * 64, 1024]
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 偏置的个数和权重的个数一致
b_fc1 = bias_variable([1024])
# 这里将第二层池化后的张量（长：7 宽：7 深度：64） 变成向量（跟上一节的Softmax模型的输入一样了）
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 使用ReLU激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 全连接层输入的大小为1024,而我们要得到的结果的大小是10（0～9），
# 所以这里权重W的尺寸为[1024, 10]
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 最后都要经过Softmax函数将输出转化为概率问题
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数和损失优化
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 测试准确率,跟Softmax回归模型的一样
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 将训练结果保存，如果不保存我们这次训练结束后的结果也随着程序运行结束而释放了
savePath = '.mnist_conv'
saveFile = savePath + 'mnist_conv.ckpt'
if os.path.exists(savePath) == False:
    os.mkdir(savePath)

saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    # 训练两万次
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        # 每训练100次，我们打印一次训练的准确率
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        # 这里是真的训练，将数据传入
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('end train, start testing...')

    mean_value = 0.0
    for i in range(mnist.test.labels.shape[0]):
        batch = mnist.test.next_batch(50)
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        mean_value += train_accuracy

    print('test accuracy %g' % (mean_value / mnist.test.labels.shape[0]))
    # 训练结束后，我们使用mnist.test在测试最后的准确率
    #print("test accuracy %g" % sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

    # 最后，将会话保存下来
    saver.save(sess, saveFile)
