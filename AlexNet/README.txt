AlexNet网络

1. AlexNet网络的构建过程：AlexNet_model程序（注释）中创建了一个类来定义AlexNet模型图，并带有加载预训练参数的函数

2.读取本地图片制作自己的数据集，包括图片预处理过程，用的是猫狗大战图片

3.利用训练集对AlexNet网络进行微调，对AlexNet网络中第六、七、八全连接层进行重新训练