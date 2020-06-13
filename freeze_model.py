# freeze_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from const import const_image_height
from const import const_image_width
from const import const_use_batch_norm
from const import const_use_kernel_regularizer

from hed_net import *
from input_pipeline import *
from util import *

import argparse

parser = argparse.ArgumentParser(description='argparse for freeze hed')

parser.add_argument('--checkpoint_dir', '-checkpoint', type=str,
                    help='checkpoint directory.', required=False)

args = parser.parse_args()

if __name__ == "__main__":
    hed_graph_without_weights_file_name = 'hed_graph_without_weights.pb'
    hed_graph_without_weights_file_path = os.path.join(args.checkpoint_dir, hed_graph_without_weights_file_name)
    hed_graph_file_path = os.path.join(args.checkpoint_dir, 'hed_graph.pb')
    hed_tflite_model_file_path = os.path.join(args.checkpoint_dir, 'hed_lite_model.tflite')

    batch_size = 1
    image_float = tf.placeholder(tf.float32, shape=(batch_size, const_image_height, const_image_width, 3),
                                 name='hed_input')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
    print('###1 image_float shape is: {}, name is: {}'.format(image_float.get_shape(), image_float.name))
    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = vgg_style_hed(image_float, batch_size, is_training_placeholder)
    print('###2 dsn_fuse shape is: {}, name is: {}'.format(dsn_fuse.get_shape(), dsn_fuse.name))

    # Saver
    hed_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    saver = tf.train.Saver(hed_weights)

    global_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_init)

        latest_ck_file = tf.train.latest_checkpoint(args.checkpoint_dir)
        if latest_ck_file:
            print('restore from latest checkpoint file: {}'.format(latest_ck_file))
            saver.restore(sess, latest_ck_file)
        else:
            print('no checkpoint file to restore, exit()')
            exit()

        print('##############################################################')
        print('##############################################################')
        print('##############################################################')
        print('Input Node is:')
        print('  %s' % image_float)
        print('  %s' % is_training_placeholder)
        print('Output Node is:')
        print('  %s' % dsn_fuse)
        print('##############################################################')
        print('##############################################################')
        print('##############################################################')

        #################################################
        #################################################
        # Write input graph pb file
        tf.train.write_graph(sess.graph.as_graph_def(), args.checkpoint_dir, hed_graph_without_weights_file_name)

        input_saver_def_path = ''
        input_binary = False
        input_checkpoint_path = latest_ck_file
        output_node_names = 'hed/dsn_fuse/conv2d/BiasAdd'
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        clear_devices = False
        # Tensorflow自带的这个freeze_graph函数，文档解释的不清楚，
        freeze_graph.freeze_graph(hed_graph_without_weights_file_path, input_saver_def_path,
                                  input_binary, input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, hed_graph_file_path,
                                  clear_devices, '')
        ################################################
        print('freeze to pb model finished')
