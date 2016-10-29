import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops


class Defaults(object):
    padding = 'SAME'
    wd = None
    
    def __init__(self, padding='SAME', wd=None, device='/cpu:0'):
        self.previous_padding = Defaults.padding
        self.previous_wd = Defaults.wd
        
        Defaults.padding = padding
        Defaults.wd = wd
        self.device = tf.device(device)
        
    def __enter__(self):
        return self.device.__enter__()
    
    def __exit__(self, type, value, traceback):
        Defaults.padding = self.previous_padding
        Defaults.wd = self.previous_wd
        return self.device.__exit__(type, value, traceback)


#https://github.com/joelthchao/tensorflow-finetune-flickr-style/blob/master/network.py

def load(data_path, session):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

def get_unique_name(prefix):        
    id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
    return '%s_%d'%(prefix, id)

def make_var(name, shape, trainable=True, wd=None):
    wd = Defaults.wd if wd is None else wd
    var = tf.get_variable(name, shape, trainable=trainable)
    if wd is not None and trainable:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=None, group=1, trainable=True):
    padding = Defaults.padding if padding is None else padding
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_var('weights', shape=[k_h, k_w, c_i/group, c_o], trainable=trainable)
        biases = make_var('biases', shape=[c_o], trainable=trainable)
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)                
        if relu:
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            return tf.tanh(bias, name=scope.name)
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)

def relu(input, name):
    return tf.nn.relu(input, name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=Defaults.padding):
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=Defaults.padding):
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

def concat(inputs, axis, name):
    return tf.concat(concat_dim=axis, values=inputs, name=name)

def fc(input, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = make_var('weights', shape=[num_in, num_out])
        biases = make_var('biases', [num_out])
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(input, weights, biases, name=scope.name)
        return fc

def softmax(input, name):
    return tf.nn.softmax(input, name)

def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

def batch_norm(x, n_out, phase_train, name='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name, initializer=tf.constant(0.0, shape=[n_out])):
        beta = make_var('beta', shape=None)
            
    with tf.variable_scope(name, initializer=tf.constant(1.0, shape=[n_out])):
        gamma = make_var('gamma', shape=None)
        
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name="batch_norm")
    return normed

def gram_matrix(input, name, scale=1.0):
    input.get_shape().assert_has_rank(4)
    
    batch, height, width, number = map(lambda i: i.value, input.get_shape())
    size = height * width * number
    feats = tf.reshape(input, (-1, number)) * scale
    gram = tf.matmul(feats, feats, transpose_a=True) / size
    return gram
