import numpy as np
import tensorflow as tf
import os

class yoloV3:
    def __init__(self,epsilon,decay,archor_path,classes_path,Train):
        self.epsilon = epsilon
        self.decay = decay
        self.archor_path = archor_path
        self.classes_path = classes_path
        self.Train = Train

        self.archors = self.get_archors()
        self.classes = self.get_classes()
    def get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            classes_name = f.readlines()
        classes_name = [c.strip() for c in classes_name]
        return classes_name
    def get_archors(self):
        archors_path = os.path.expanduser(self.archor_path)
        with open(archors_path) as a:
            archors = a.readlines()
        archors = [float(x) for x in archors.split(',')]
        return np.array(archors).reshape(-1,2)

    # ------------------------------------------------------------------------------------------------------------------
    # 封装了一个BN层
    def bn_layers(self,inputlayer,name=None,Train=True,decay=0.99,epsilon=1e-3):
        bn_layer = tf.layers.batch_normalization(inputs=inputlayer,momentum=decay,epsilon=epsilon,center=True,scale=True,training=Train,name=name)
        return tf.nn.leaky_relu(bn_layer,alpha=0.1)
    # 封装conv卷积
    def conv_layers(self,inputs,filter_num,kernel_size,name,use_bias=False,strides=1):
        conv_layer = tf.layers.conv2d(inputs,filters=filter_num,kernel_size=kernel_size,strides=[strides,strides],
                                      padding=('same' if strides==1 else 'valid'),use_bias=use_bias,
                                      kernel_initializer=tf.glorot_uniform_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regulazier(5e-4))
        return conv_layer
    # 封装残差块----3x3→(1x1→3x3)
    def res_block(self,inputs,filter_num,block_num,conv_index,training=True,decay=0.99,epsilon=1e-3):
        # 先填充，不明觉厉
        inputs = tf.pad(inputs,paddings=[[0,0],[1,0],[1,0],[0,0]],mode='constant')
        layer = self.conv_layers(inputs,filter_num,kernel_size=3,strides=2,name='conv2d'+str(conv_index))
        layer = self.bn_layers(layer,name='bn'+str(conv_index),training=training,decay=decay,epsilon=epsilon)
        conv_index+=1
        for i in range(block_num):
            shortcut = layer
            layer = self.conv_layers(layer,filter_num=filter_num//2,kernel_size=1,strides=1,name='conv2d'+str(conv_index))
            layer = self.bn_layers(layer,name='bn'+str(conv_index),training=training,epsion=epsilon,decay=decay)
            conv_index += 1
            layer = self.conv_layers(layer, filter_num=filter_num, kernel_size=3, strides=1,
                                     name='conv2d' + str(conv_index))
            layer = self.bn_layers(layer, name='bn' + str(conv_index), training=training, epsion=epsilon, decay=decay)
            conv_index += 1
            layer +=shortcut
        return layer,conv_index

    # ------------------------------------------------------------------------------------------------------------------
    # darknet-53
    def darknet53(self,inputs,conv_index,training=True,decay=0.99,epsilon=1e-3):
        # 416x416x3→13x13x1024
        with tf.variable_scope('darknet53'):
            conv = self.conv_layers(inputs,filter_num=32,kernel_size=3,strides=1,name='conv'+str(conv_index))
            conv = self.bn_layers(conv,name='bn'+str(conv_index),training=training,decay=decay,epsilon=epsilon)
            conv_index+=1
            conv,conv_index = self.res_block(conv,conv_index=conv_index,filter_num=64,block_num=1,training=training,decay=decay,epsilon=epsilon)
            conv,conv_index = self.res_block(conv,conv_index=conv_index,filter_num=128,block_num=2,training=training,decay=decay,epsilon=epsilon)
            conv, conv_index = self.res_block(conv, conv_index=conv_index, filter_num=256, block_num=8,
                                              training=training, decay=decay, epsilon=epsilon)
            route1 = conv
            conv, conv_index = self.res_block(conv, conv_index=conv_index, filter_num=512, block_num=8,
                                              training=training, decay=decay, epsilon=epsilon)
            route2 = conv
            conv, conv_index = self.res_block(conv, conv_index=conv_index, filter_num=1024, block_num=4,
                                              training=training, decay=decay, epsilon=epsilon)
            return route1,route2,conv,conv_index

    # ------------------------------------------------------------------------------------------------------------------
    # yolo3net
    def yolo3net(self,inputs,in_filters,out_filters,conv_index,training=True,decay=0.99,epsilon=1e-3):
        conv = self.conv_layers(inputs, filter_num=in_filters, kernel_size=1, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        conv = self.conv_layers(inputs, filter_num=in_filters*2, kernel_size=3, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        conv = self.conv_layers(inputs, filter_num=in_filters, kernel_size=1, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        conv = self.conv_layers(inputs, filter_num=in_filters*2, kernel_size=3, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        conv = self.conv_layers(inputs, filter_num=in_filters, kernel_size=1, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        route = conv

        conv = self.conv_layers(inputs, filter_num=in_filters*2, kernel_size=3, strides=1, name='conv' + str(conv_index))
        conv = self.bn_layers(conv, name='bn' + str(conv_index), training=training, decay=decay, epsilon=epsilon)
        conv_index += 1
        conv = self.conv_layers(inputs, filter_num=out_filters, kernel_size=1, strides=1, name='conv' + str(conv_index))
        conv_index += 1
        return route,conv,conv_index
    def yolo3inference(self,inputs,num_anchors,num_classes,training=True):
        conv_index = 1
        conv26,conv43,conv,conv_index = self.darknet53(inputs,conv_index,training=training,decay=self.decay,epsilon=self.epsilon)
        with tf.variable_scope('yolo'):
            conv57,conv59,conv,conv_index = self.yolo3net(conv,512,num_anchors*(num_classes+5),conv_index=conv_index,training=training,decay=self.decay,epsilon=self.epsilon)
            conv60 = self.conv_layers(conv57,filter_num=256,kernel_size=1,strides=1,name='conv'+conv_index)
            conv60 = self.bn_layers(conv60,name='bn'+conv_index,training=training,decay=self.decay,epsilon=self.epsilon)
            conv_index+=1
            # 上采样？？？？
            upsample0 = tf.image.resize_nearest_neighbor(conv60,[2*tf.shape(conv60)[1],2*tf.shape(conv60)[1]],name='upsample0')
            route0 = tf.concat([upsample0,conv43],axis=-1,name='route0')
            conv65, conv67, conv, conv_index = self.yolo3net(conv, 256, num_anchors * (num_classes + 5),
                                                             conv_index=conv_index, training=training, decay=self.decay,
                                                             epsilon=self.epsilon)

            # ????
            conv68 = self.conv_layers(conv65, filter_num=128, kernel_size=1, strides=1, name='conv' + conv_index)
            conv68 = self.bn_layers(conv68, name='bn' + conv_index, training=training, decay=self.decay,
                                    epsilon=self.epsilon)
            conv_index += 1
            # 上采样？？？？
            upsample1 = tf.image.resize_nearest_neighbor(conv68, [2 * tf.shape(conv68)[1], 2 * tf.shape(conv68)[1]],
                                                         name='upsample0')
            route1 = tf.concat([upsample1, conv26], axis=-1, name='route1')
            _,conv75,_ = self.yolo3net(route1, 128, num_anchors * (num_classes + 5),
                                                             conv_index=conv_index, training=training, decay=self.decay,
                                                             epsilon=self.epsilon)
            return [conv59,conv67,conv75]







