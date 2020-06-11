from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tf
from const import const_image_height
from const import const_image_width
from const import const_use_batch_norm
from const import const_use_kernel_regularizer


def class_balanced_sigmoid_cross_entropy(logits, label):
    ##ref - https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py

    '''
    The class-balanced cross entropy loss,
    as in 'Holistically-Nested Edge Detection <http://arxiv.org/abs/1504.06375>'
    Args:
      logits: of shape(b, ...).
      label: of the same shape. the ground truth is {0,1}
    Returns:
      class-balanced cross entropy loss.
    '''

    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        count_neg = tf.reduce_sum(1.0 - label)  # 样本中0的数量
        count_pos = tf.reduce_sum(label)  # 样本中1的数量
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1.0 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))

        # 如果样本中1的数量等于0，直接就让cost为0，因为beta == 1时，pos_weight=beta/(1-beta)的结果是无穷大
        zero = tf.equal(count_pos, 0.0)
        final_cost = tf.where(zero, 0.0, cost)
    return final_cost


def vgg_style_hed(inputs, batch_size, is_training):
    filter_initializer = tf.contrib.layers.xavier_initializer()
    if const_use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
    else:
        weights_regularizer = None

    def _vgg_conv2d(inputs, filters, kernel_size):
        use_bias = True
        if const_use_batch_norm:
            use_bias = False

        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   padding='same',
                                   activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        if const_use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = tf.nn.relu(outputs)
        return outputs

    def _max_pool2d(inputs):
        outputs = tf.layers.max_pooling2d(inputs,
                                          [2, 2],
                                          strides=(2, 2),
                                          padding='same')
        return outputs

    def _dsn_1x1_conv2d(inputs, filters):
        use_bias = True
        if const_use_batch_norm:
            use_bias = False

        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   padding='same',
                                   activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        if const_use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        ##no activation

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   padding='same',
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        ##no batch normalization
        ##no activation
        return outputs

    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters,
                                             kernel_size,
                                             strides=(upsample_factor, upsample_factor),
                                             padding='same',
                                             activation=None,
                                             use_bias=True,
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)
        return outputs

    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        with tf.variable_scope('conv1'):
            net = _vgg_conv2d(net, 12, [3, 3])
            net = _vgg_conv2d(net, 12, [3, 3])
            dsn1 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv2'):
            net = _vgg_conv2d(net, 24, [3, 3])
            net = _vgg_conv2d(net, 24, [3, 3])
            dsn2 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv3'):
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            dsn3 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv4'):
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            dsn4 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv5'):
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            dsn5 = net
            # no need this pool layer

        ##dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            print('!!debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            print('!!debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor=2)
            print('!!debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            print('!!debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor=4)
            print('!!debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            print('!!debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor=8)
            print('!!debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            print('!!debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor=16)
            print('!!debug, dsn5 shape is: {}'.format(dsn5.get_shape()))

        ##dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            print('!!debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            print('!!debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5