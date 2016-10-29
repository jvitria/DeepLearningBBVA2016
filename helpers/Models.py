import tensorflow as tf
import numpy as np
import sys
from Network import *

def vgg16_small(x, trainable=False):
    # Conv 1
    conv1_1 = conv(x, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1', trainable=trainable)
    conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2', trainable=trainable)
    pool1 = max_pool(conv1_2, 3, 3, 2, 2, padding='SAME', name='pool1')
    
    #norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
    
    # Conv 2
    conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1', trainable=trainable)
    conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2', trainable=trainable)
    pool2 = max_pool(conv2_2, 3, 3, 2, 2, padding='SAME', name='pool2')
    
    #norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    
    # Conv 3
    conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1', trainable=trainable)
    conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2', trainable=trainable)
    #conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
    pool3 = max_pool(conv3_2, 3, 3, 2, 2, padding='SAME', name='pool3')
    
    # Conv 4
    conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1', trainable=trainable)
    conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2', trainable=trainable)
    #conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
    pool4 = max_pool(conv4_2, 3, 3, 2, 2, padding='SAME', name='pool4')
    
    # Conv 4
    conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1', trainable=trainable)
    conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2', trainable=trainable)
    #conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
    
    return conv5_2, {'conv4_2': conv4_2, 'conv4_1': conv4_1,
                     'conv3_2': conv3_2, 'conv3_1': conv3_1,
                     'conv2_2': conv2_2, 'conv2_1': conv2_1,
                     'conv1_2': conv1_2, 'conv1_1': conv1_1}

def vgg16(x):
    # Conv 1
    conv1_1 = conv(x, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
    conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
    pool1 = max_pool(conv1_2, 3, 3, 2, 2, padding='SAME', name='pool1')
    
    #norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
    
    # Conv 2
    conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
    conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
    pool2 = max_pool(conv2_2, 3, 3, 2, 2, padding='SAME', name='pool2')
    
    #norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    
    # Conv 3
    conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
    conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
    conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
    pool3 = max_pool(conv3_3, 3, 3, 2, 2, padding='SAME', name='pool3')
    
    # Conv 4
    conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
    conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
    conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
    pool4 = max_pool(conv4_3, 3, 3, 2, 2, padding='SAME', name='pool4')
    
    # Conv 4
    conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
    conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
    conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
    
    return conv5_3, {'conv4_3': conv4_3, 'conv4_2': conv4_2, 'conv4_1': conv4_1,
                     'conv3_3': conv3_3, 'conv3_2': conv3_2, 'conv3_1': conv3_1,
                     'conv2_2': conv2_2, 'conv2_1': conv2_1,
                     'conv1_2': conv1_2, 'conv1_1': conv1_1}



