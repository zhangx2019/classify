# -*- coding: utf-8 -*-

#利用Tensorflow对预训练的AlexNet网络进行微调
import tensorflow as tf
import numpy as np
import os
from AlexNet_model import AlexNet
#from datagenerator import ImageDataGenerator
#from datetime import datetime
#from tensorflow.contrib.data import Iterator
import input_data

#模型保存的路径和文件名
MODEL_SAVE_PATH="./model/"
MODEL_NAME="alexnet_model.ckpt"

#训练集图片所在路径
train_dir='D:/coding/Image Classify/AlexNet/train/'
#训练图片的尺寸
image_size=227
#训练集中图片总数
total_size=25000

#学习率
learning_rate=0.001
#训练完整数据集迭代轮数
num_epochs=10
#数据块大小
batch_size=128

#执行Dropout操作所需的概率值
dropout_rate=0.5
#类别数目
num_classes=2
#需要重新训练的层
train_layers=['fc8','fc7','fc6']

#读取本地图片，制作自己的训练集，返回image_batch，label_batch
train, train_label = input_data.get_files(train_dir)
x,y=input_data.get_batch(train,train_label,image_size,image_size,batch_size,2000)

#用于计算图输入和输出的TF占位符，每次读取一小部分数据作为当前的训练数据来执行反向传播算法
#x =tf.placeholder(tf.float32,[batch_size,227,227,3],name='x-input')
#y =tf.placeholder(tf.float32,[batch_size,num_classes])
keep_prob=tf.placeholder(tf.float32)

#定义神经网络结构，初始化模型
model =AlexNet(x,keep_prob,num_classes,train_layers)
#获得神经网络前向传播的输出
score=model.fc8

#获得想要训练的层的可训练变量列表
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

#定义损失函数，获得loss
with tf.name_scope("cross_ent"):
	loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,labels=y))

#定义反向传播算法（优化算法）
with tf.name_scope("train"):
    # 获得所有可训练变量的梯度
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # 选择优化算法，对可训练变量应用梯度下降算法更新变量
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

#使用前向传播的结果计算正确率
with tf.name_scope("accuracy"):
    correct_pred=tf.equal(tf.cast(tf.argmax(score,1),tf.int32),y)
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initialize an saver for store model checkpoints 加载模型
saver=tf.train.Saver()

# 每个epoch中验证集/测试集需要训练迭代的轮数，np.floor返回不大于输入参数的最大整数，向下取整
train_batches_per_epoch = int(np.floor(total_size/batch_size))

with tf.Session() as sess:
    #变量初始化
    tf.global_variables_initializer().run()

    #tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    #tf.train.start_queue_runners, 启动入队线程，由多个或单个线程，按照设定规则，把文件读入FilenameQueue中。
    #函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for epoch in range(num_epochs):
            #训练完一个epoch所需的迭代次数
            for step in range(train_batches_per_epoch):
                #coord.should_stop()来查询是否应该终止所有线程
                if coord.should_stop():
                    break
                #sess.run 来启动数据出列和执行计算
                _,loss_value,accu=sess.run([train_op,loss,accuracy],feed_dict={keep_prob: 1.})
                if step%50==0:
                    print("Afetr %d training step(s),loss on training batch is %g,accuracy is %g." % (step,loss_value,accu))

        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
    #当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        #coord.request_stop()来发出终止所有线程的命令
        coord.request_stop()
    #coord.join(threads)把线程加入主线程，等待threads结束
    coord.join(threads)
