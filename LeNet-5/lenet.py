import tensorflow as tf
import numpy, pickle
# import matplotlib.pyplot as plt

# 1.处理输入
# cifar10中训练batch存储格式为字典,其中
# data:10000个32*32*3图像,labels:0-9数字表示10个类别,filenames:10000个图像名称,batch_label为batch序号
# 输出为X(10000, 3072), Y(10000, 10)
def load_cifar_10_data_batch(filename):
    file = open(filename, 'rb')
    datadictionary = pickle.load(file, encoding='latin1')
    X = datadictionary['data'] # X.shape = (1000, 3072)
    Y = datadictionary['labels']
    # filenames = datadictionary['filenames']
    # batch_label = datadictionary['batch_label']
    # X = X.reshape(10000, 3, 32, 32).transpose(0, 2 ,3 ,1).astype('float')
    Y_temp = numpy.array(Y)
    return X, Y
path = '/coding/Image Classify/LeNet-5'
X_train, Y_train = load_cifar_10_data_batch(path + '/cifar-10-batches-py/data_batch_1')
X_test, Y_test = load_cifar_10_data_batch(path + '/cifar-10-batches-py/test_batch')

# 2.构造网络所需的函数
# 2.1 参数w初始化, tf.random_normal为根据shape产生随机分布,均值标准差自定, shape=[height, width, channels, filters]
def weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, mean=0, stddev=0.1))
# 2.2 参数b初始化, tf.constant为根据shape产生常量
def biases(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
# 2.3 卷积函数, x=[m, height, width, channels], m为待训练集数量
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=True)
# 2.4 池化函数, 池化其实也相当于一次卷积, 比较特殊的卷积
def max_pool(x):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 2.5 计算准确率, 根据测试集
def compute_accuracy(x_test, y_test):
    global predict_function
    predict_test = sess.run(predict_function, feed_dict={xs:x_test})
    # 比较预测值与测试集最大值下标是否相同, 返回相同个数
    y_test_one_hot = tf.one_hot(y_test, depth = 10)
    correct_num = tf.equal(tf.argmax(predict_test, 1), tf.argmax(y_test_one_hot, 1))
    # 计算均值即为精确率, 并转格式float32
    accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))
    return sess.run(accuracy, feed_dict={xs:x_test, ys:y_test})

# 3.定义输入, xs为训练集输入m*3072矩阵, ys为训练集labelsm*1矩阵
xs = tf.placeholder(tf.float32, [None, 32*32*3])/255 #/255，将图像矩阵转换到0-1之间
ys = tf.placeholder(tf.int32, [None])

# 4. 构造网络
# x0为输入m*32*32*3, 图像数量m, 大小32*32, channels为3
x0 = tf.transpose(tf.reshape(xs, [-1, 3, 32, 32]), perm=[0, 2, 3, 1])
# 4.1 构建卷积层, 卷积、池化、再卷积、再池化、最后拉伸
W_conv1 = weights([5, 5, 3, 20])
b_conv1 = biases([20])
a_conv1 = conv2d(x0, W_conv1) + b_conv1
z_conv1 = tf.nn.relu(a_conv1)
h_conv1 = max_pool(z_conv1)

W_conv2 = weights([5, 5, 20, 50])
b_conv2 = biases([50])
a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
z_conv2 = tf.nn.relu(a_conv2)
h_conv2 = max_pool(z_conv2)

# x_conv0为卷积池化后拉伸的输入m*(8*8*35=2240), 图像数量m, 输入为2240
x_conv0 = tf.reshape(h_conv2, [-1, 8*8*50])
# 4.2 接入fully connected networks, 网络各层单元数为2240, 1024, 10
W_fc1 = weights([8*8*50, 1024])
b_fc1 = biases([1024])
a_fc1 = tf.matmul(x_conv0, W_fc1) + b_fc1
z_fc1 = tf.nn.sigmoid(a_fc1)

W_fc2 = weights([1024, 10])
b_fc2 = biases([10])
a_fc2 = tf.matmul(z_fc1, W_fc2) + b_fc2
# z_fc2 = tf.nn.relu(a_fc2)


# 4.3 预测函数与代价函数, 采用最简单的
predict_function = a_fc2
ys_one_hot = tf.one_hot(ys, depth = 10)
# cost_function = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict_function), reduction_indices=[1]))
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=predict_function))

# 4.4 梯度下降, 使cost_function值最小
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cost_function)
# train_step = tf.train.AdamOptimizer(1e-3).minimize(cost_function)

# 5. 训练并图像化表示
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
for i in range(20):
    for t in range(0, 10000-32, 32):
        iter = t // 32
        xs_batch, ys_batch = X_train[t:t+32], Y_train[t:t+32]
        sess.run(train_step, feed_dict={xs:xs_batch, ys:ys_batch})
        if iter % 10 == 0:
            cost = sess.run(cost_function, feed_dict={xs: xs_batch, ys: ys_batch})
            accuracy = compute_accuracy(X_test[:10000], Y_test[:10000])
            print("iters:%s, cost:%s, accuracy:%s" % (iter, cost, accuracy))
# saver.save(sess, '/Users/wanglei/Downloads/test/model.ckpt')
sess.close()
