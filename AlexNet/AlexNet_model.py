# -*- coding: utf-8 -*-

#定义AlexNet神经网络结构模型
import tensorflow as tf
import numpy as np

#建立模型图
class AlexNet(object):

    def __init__(self,x,keep_prob,num_classes,skip_layer,weights_path='DEFAULT'):
        self.X=x # 输入图片
        self.NUM_CLASSES = num_classes # num_classes:数据类别数
        self.KEEP_PROB = keep_prob  # keep_prob:dropout概率
        self.SKIP_LAYER = skip_layer  # 预训练参数未赋值给self.SKIP_LAYER中指定的网络层
        if weights_path=='DEFAULT':
            self.WEIGHT_PATH='D:/coding/Image Classify/AlexNet/bvlc_alexnet.npy'
        else:
            self.WEIGHT_PATH=weights_path

        self.alexnet()

    def alexnet(self):
        # 卷积层与全连接层激活函数用的是Relu，防止使用sigmoid在网络较深时的梯度消失
        # 第一层：卷积层-->最大池化层-->LRN
        conv1=conv_layer(self.X,11,11,96,4,4,padding='VALID',name='conv1') # (227-11)/4+1=55,55*55*96
        self.conv1=conv1
        pool1=max_pool(conv1,3,3,2,2,padding='VALID',name='pool1') # (55-3)/2+1=27,27*27*96
        # 局部响应归一化作用：使得神经元响应比较大得值变得更大，抑制其他反馈较小得神经元
        # 增强了模型的泛化能力
        norm1=lrn(pool1,2,2e-05,0.75,name='norml')

        # 第二层：卷积层-->最大池化层-->LRN
        conv2=conv_layer(norm1,5,5,256,1,1,groups=2,name='conv2') # 扩展2个像素，即(27-5+2*2)/1+1=27,27*27*256,分为2组
        self.conv2=conv2
        pool2=max_pool(conv2,3,3,2,2,padding='VALID',name='pool2') # (27-3)/2+1=13,13*13*256
        norm2=lrn(pool2,2,2e-05,0.75,name='norm2')

        # 第三层：卷积层
        conv3=conv_layer(norm2,3,3,384,1,1,name='conv3') # 扩展1个像素，即(13-3+1*2)/1+1=13,13*13*384
        self.conv3=conv3

        #第四层：卷积层
        conv4=conv_layer(conv3,3,3,384,1,1,groups=2,name='conv4') # 扩展1个像素，即(13-3+1*2)/1+1=13,13*13*384
        self.conv4=conv4

        #第五层：卷积层-->最大池化层
        conv5=conv_layer(conv4,3,3,256,1,1,groups=2,name='conv5') # 扩展1个像素，即(13-3+1*2)/1+1=13,13*13*256
        self.conv5=conv5
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5') # (13-3)/2+1=6,6*6*256

        # 第六层：全连接层
        #def reshape(tensor, shape, name=None)
        # 第1个参数为被调整维度的张量。
        # 第2个参数为要调整为的形状。
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])  # 6*6*256=9216
        fc6 = fc_layer(flattened, 6 * 6 * 256, 4096, name='fc6')
        # 随机忽略一部分神经元，以避免模型过拟合
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 第七层：全连接层
        fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 第八层：全连接层，不带激活函数
        self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    # 加载神经网络预训练参数,将存储于self.WEIGHTS_PATH的预训练参数赋值给那些没有在self.SKIP_LAYER中指定的网络层的参数
    def load_initial_weights(self,session):
        #下载权重文件npy为二进制文件,np（numpy）专用的npy(二进制格式)或npz(压缩打包格式)格式
        '''
        np.save("a.npy", a.reshape(3,4))
        c = np.load("a.npy")
        c
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        a = np.array([[1,2,3],[4,5,6]])
        b = np.arange(0,1.0,0.1)
        c = np.sin(b)
        np.savez("result.npz", a, b, sin_arr=c)  #使用sin_arr命名数组c
        r = np.load("result.npz") #加载一次即可
        r["arr_0"]
        array([[1, 2, 3],
               [4, 5, 6]])
        r["arr_1"]
        array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
        r["sin_arr"]
        array([ 0.        ,  0.09983342,  0.19866933,  0.29552021,  0.38941834,
               0.47942554,  0.56464247,  0.64421769,  0.71735609,  0.78332691])
        '''
        #item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
        # person = {'name': 'lizhong', 'age': '26', 'city': 'BeiJing', 'blog': 'www.jb51.net'}
        #
        # for x in person.items():
        #     print (x) #('name','lizhong')等等
        weights_dict=np.load(self.WEIGHTS_PATH,encoding='bytes').item()
        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                #True: 参数空间使用reuse 模式，即该空间下的所有tf.get_variable()函数将直接获取已经创建的变量，
                # 如果参数不存在tf.get_variable()函数将会报错
                #AUTO_REUSE：若参数空间的参数不存在就创建他们，如果已经存在就直接获取它们。
                #None 或者False 这里创建函数tf.get_variable()函数只能创建新的变量，当同名变量已经存在时，函数就报错
                with tf.variable_scope(op_name,reuse=True):
                    for data in weights_dict[op_name]:
                        #偏置项
                        if len(data.shape)==1:
                            var=tf.get_variable('biases',trainable=False)
                            session.run(var.assign(data))  #偏置赋值
                        #权重
                        else:
                            var=tf.get_variable('weights',trainable=False)
                            session.run(var.assign(data))  #权重赋值

# 定义卷积层，当groups=1时，AlexNet网络不拆分；当groups=2时，AlexNet网络拆分成上下两个部分
def conv_layer(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME',
                   groups=1):
    # 获得输入图像的通道数
    input_channels=int(x.get_shape()[-1])

    #创建lambda表达式
    convovle=lambda i,k:tf.nn.conv2d(i,k,strides=[1,stride_y,stride_x,1],padding=padding)

    with tf.variable_scope(name) as scope:
        #创建卷积层所需的权重参数和偏置项参数
        weights=tf.get_variable("weights",shape=[filter_height,filter_width,input_channels/groups,num_filters])
        biases=tf.get_variable("biases",shape=[num_filters])

    if groups==1:
        conv=convovle(x,weights)
    #当groups不等于1时，拆分输入和权重
    else:
        input_groups=tf.split(axis=3,num_or_size_splits=groups,value=x)
        weight_groups=tf.split(axis=3,num_or_size_splits=groups,value=weights)
        output_groups=[convovle(i,k) for i,k in zip(input_groups,weight_groups)]
        #单独计算完后，再次根据深度连接两个网络
        conv=tf.concat(axis=3,values=output_groups)

    #加上偏置项
    bias=tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape().as_list())
    #激活函数
    relu=tf.nn.relu(bias,name=scope.name)

    return relu

# 定义全连接层
def fc_layer(x,num_in,num_out,name,relu=True):
    with tf.variable_scope(name) as scope:
        #创建权重参数和偏置项
        weights=tf.get_variable("weights",shape=[num_in,num_out],trainable=True)
        biases=tf.get_variable("biases",[num_out],trainable=True)

        #计算
        act=tf.nn.xw_plus_b(x,weights,biases,name=scope.name)

        if relu==True:
            relu=tf.nn.relu(act)
            return relu
        else:
            return act

#定义最大池化层
def max_pool(x,filter_height,filter_width,stride_y,stride_x,name,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding,name=name)

#定义局部响应归一化LPN
def lrn(x,radius,alpha,beta,name,bias=1.0):
    return tf.nn.local_response_normalization(x,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)

#定义dropout
def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)





