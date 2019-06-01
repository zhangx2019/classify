# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

#生成训练图片的路径
train_dir='D:/coding/Image Classify/AlexNet/train/'

#获取图片，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir):

    cats =[]
    label_cats =[]
    dogs =[]
    label_dogs =[]
    '''
    os.listdir(path=''),其中参数path为需要列出的目录路径。该函数返回指定的文件夹包含的文件或文件夹的名字的列表。
    test_file:
         test1.txt
         test2.txt
         test3.txt
    >>> import os
    >>> path = r'C:/Users/XXN/Desktop/test_file'
    >>> for each_file in os.listdir(path):	
            print(os.path.join(path,each_file))
    结果如下：C:/Users/XXN/Desktop/test_file/test1.txt
              C:/Users/XXN/Desktop/test_file/test2.txt
              C:/Users/XXN/Desktop/test_file/test3.txt
    当一个目录下面既有文件又有目录（文件夹），可使用os.walk()读取里面所有文件。
    Test_file：        
        file1:             
            test1.txt            
            test2.txt             
            test3.txt        
        file2:            
            test1.txt             
            test2.txt             
            test3.txt        
        test1.txt        
        test2.txt        
        test3.txt
    >>> import os
    >>> path = r'C:/Users/XXN/Desktop/test_file'
    >>> for parent,dirnames,filenames in os.walk(path):	
            print(parent,dirnames,filenames) 
    结果如下：C:/Users/XXN/Desktop/test_file ['file1', 'file2'] ['test1.txt', 'test2.txt', 'test3.txt']
              C:/Users/XXN/Desktop/test_file/file1 [] ['test1.txt', 'test2.txt', 'test3.txt']
              C:/Users/XXN/Desktop/test_file/file2 [] ['test1.txt', 'test2.txt', 'test3.txt']
    '''
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)  #图片存放到对应的列表中
            label_cats.append(0)          #标签存放到label列表中
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    #合并数据
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    #利用shuffle打乱数据
    temp = np.array([image_list, label_list])
    temp = temp.transpose()# 转置
    np.random.shuffle(temp)

    #将所有的image和label转换成list
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

#将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
def get_batch(image,label,image_W,image_H,batch_size,capacity):

    #将python.list类型转换成tf能够识别的格式，tf.cast转换格式
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    #产生一个输入队列queue
    #tf.train.slice_input_producer从本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中
    input_queue=tf.train.slice_input_producer([image,label])

    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    #将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image=tf.image.decode_jpeg(image_contents,channels=3)

    #将数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)

    #生成batch，一个batch包括的image_batch与label_batch
    #tf.train.batch从文件名队列中提取tensor，使用单个或多个线程，准备放入文件队列
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)

    #重新排列标签，行数为[batch_size]
    #label_batch=tf.reshape(label_batch,[batch_size])
    image_batch=tf.cast(image_batch,tf.float32)

    return image_batch,label_batch


