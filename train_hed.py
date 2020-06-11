from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import tensorflow as tf
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion('1.6'), 'please use tensorflow version 1.6 or newer'
print('Tensorflow Version: {}'.format(tf.__version__))

import argparse
import os

from const import const_image_height
from const import const_image_width
from const import const_use_batch_norm
from const import const_use_kernel_regularizer

from hed_net import *
from input_pipeline import *
from util import *

parser = argparse.ArgumentParser(description='argparse for HED')

parser.add_argument('--dataset_root_dir', '-dataset', type=str,
                    help='Root directory to put all the training data.', required=False)
parser.add_argument('--csv_path', '-csv', type=str,
                    help='CSV file path.', required=False)
parser.add_argument('--checkpoint_dir', '-checkpoint', type=str, default='./checkpoint',
                    help='Checkpoint directory.', required=False)
parser.add_argument('--debug_image_dir', '-debugdir', type=str, default='./debug_output_image',
                    help='Debug output image directory.', required=False)
parser.add_argument('--log_dir', '-logdir', type=str, default='./log',
                    help='Log directory for tensorflow.', required=False)
parser.add_argument('--batch_size', '-batchsize', type=int, default=4,
                    help='Batch size, default 4.', required=False)
parser.add_argument('--iterations', type=int, default=10000000,
                    help='Number of interations default 10000000.', required=False)
parser.add_argument('--display_step', '-display', type=int, default=20,
                    help='Number of iterations between optimizer print info and save test image, default 20.',
                    required=False)
parser.add_argument('--learning_rate', '-learnrate', type=float, default=0.0005,
                    help='Learning rate, default 0.0005.', required=False)
parser.add_argument('--restore_checkpoint', '-restorepoint', type=bool, default=True,
                    help='If true, restore from latest checkpoint, default True.', required=False)
parser.add_argument('--just_set_batch_size_to_one', '-setbatchone', type=bool, default=False,
                    help='If true, just set batch size to one and exit(in order to call python freeze_model.py), default False.',
                    required=False)

args = parser.parse_args()

if args.dataset_root_dir == '':
    print('must set dataset_root_dir')
    exit()
if args.csv_path == '':
    print('must set csv_path')
    exit()
if args.just_set_batch_size_to_one:
    args.batch_size = 1

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.debug_image_dir):
    os.makedirs(args.debug_image_dir)
hed_ckpt_file_path = os.path.join(args.checkpoint_dir, 'hed.ckpt')
print('###################################')
print('###################################')
print('dataset_root_dir is: {}'.format(args.dataset_root_dir))
print('os.path.join(args.dataset_root_dir, \'\') is {}'.format(os.path.join(args.dataset_root_dir, '')))
print('csv_path is: {}'.format(args.csv_path))
print('checkpoint_dir is: {}'.format(args.checkpoint_dir))
print('debug_image_dir is: {}'.format(args.debug_image_dir))
print('###################################')
print('###################################')

if __name__ == "__main__":
    # 命令行传入的路径参数，不带最后的'/'，这里要把'/'补全，然后传入给fix_size_image_pipeline
    dataset_root_dir_string = os.path.join(args.dataset_root_dir, '')

    '''
    严格来说，在机器学习任务，应该区分训练集和验证集
    但是在这份代码中，因为训练样本都是合成出来的，所以就没有区分验证集了，
    而是直接通过肉眼观察args.debug_image_dir目录里输出的debug image来判断是否可以结束训练，
    然后直接放到真实的使用环境里判断模型的实际使用效果。
  
    另外，这个任务里面，评估训练效果的时候，也没有计算label和output之间的IOU值，原因如下：
    我用执行Semantic Segementation任务的UNET网络也尝试过做这个边缘检测任务，
    在这个合成的训练样本上，UNET的IOU值是远好于HED网络的，
    但是在真实使用的场景里，UNET的效果则不如HED了，
    HED检测的边缘线是有"过剩"的部分，比如边缘线比样本中的边缘线更粗、同时还会检测到一些干扰边缘线，
    这些"过剩"的部分，可以借助后续流程里的找点算法逐层过滤掉，
    而UNET的效果就正好相反了，边缘线有些时候会遇到"缺失"，而且可能会缺失掉关键的部分，比如矩形区域的拐角处，
    这种关键部分的"缺失"，在后续的找点算法里就有点无能为力。
    '''
    input_queue_for_train = tf.train.string_input_producer([args.csv_path])
    image_tensor, annotation_tensor, \
    image_path_tensor = fix_size_image_pipeline(dataset_root_dir_string, input_queue_for_train, args.batch_size)

    '''
    #常规情况下的代码，这里还应该有一个读取verify数据的pipeline
    input_queue_for_verify = tf.train.string_input_producer([args.validation_data_file_path])
    image_tensor_for_verify, annotation_tensor_for_verify,\
      image_path_tensor_for_verify = fix_size_image_pipeline(dataset_root_dir_string, input_queue_for_verify, args.batch_size)
    '''

    is_training_placeholder = tf.placeholder(tf.bool, name='is training')
    feed_dict_to_use = {is_training_placeholder: True}

    print('image_tensor shape is: {}'.format(image_tensor.get_shape()))
    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = vgg_style_hed(image_tensor,
                                                           args.batch_size,
                                                           is_training_placeholder)
    print('dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    cost = class_balanced_sigmoid_cross_entropy(dsn_fuse, annotation_tensor)

    if const_use_kernel_regularizer:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        '''
        for reg_loss in reg_losses:
          print('reg_loss shape is: {}'.format(reg_loss.get_shape()))
        '''
        reg_constant = 0.0001
        cost = cost + reg_constant * sum(reg_losses)

    print('cost shape is: {}'.format(cost.get_shape()))
    cost_reduce_mean = tf.reduce_mean(cost)  # for tf.summary

    with tf.variable_scope("adam_vars"):
        if const_use_batch_norm == True:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)

    global_init = tf.global_variables_initializer()

    # summary
    tf.summary.scalar('loss', cost_reduce_mean)
    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(args.log_dir)

    # saver
    hed_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    all_variables_can_restore = hed_weights  # 还可以加上其他的var，整体上就是【】数组
    print('#########################################')
    print('#########################################')
    print('#########################################')
    print('all_variables_can_restore are:')
    for tensor_var in all_variables_can_restore:
        print(' %r' % (tensor_var))
    print('#########################################')
    print('#########################################')
    print('#########################################')
    ckpt_saver = tf.train.Saver(all_variables_can_restore)

    print('\n\n')
    print('#####################################################################')

    with tf.Session() as sess:
        sess.run(global_init)

        if args.restore_checkpoint:
            latest_ck_file = tf.train.latest_checkpoint(args.checkpoint_dir)
            if latest_ck_file:
                print('restore from latest checkpoint file：{}'.format(latest_ck_file))
                ckpt_saver.restore(sess, latest_ck_file)
            else:
                print('no checkpoint to restore')
        else:
            print('no checkpoint to restore')

        ################################
        if args.just_set_batch_size_to_one:
            ckpt_saver.save(sess, hed_ckpt_file_path, global_step=0)
            exit()
        ################################

        print('\nStart train.....')
        print('batch_size is: {}'.format(args.batch_size))
        print('iterations is: {}'.format(args.iterations))
        print('display-step is: {}'.format(args.display_step))
        print('learning-rate is: {}'.format(args.learning_rate))
        if const_use_kernel_regularizer:
            print('++ use L2 regularizer')
        if const_use_batch_norm == True:
            print('++ use batch norm')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(args.iterations):
            feed_dict_to_use[is_training_placeholder] = True
            loss_mean, loss, summary_string = sess.run([cost_reduce_mean, cost, merged_summary_op],
                                                       feed_dict=feed_dict_to_use)
            sess.run(train_step, feed_dict=feed_dict_to_use)

            summary_string_writer.add_summary(summary_string, step)

            if step % args.display_step == 0:
                ckpt_saver.save(sess, hed_ckpt_file_path, global_step=step)

                feed_dict_to_use[is_training_placeholder] = False

                _input_image_path, _input_annotation, \
                _loss_mean, \
                _dsn_fuse, \
                _dsn1, \
                _dsn2, \
                _dsn3, \
                _dsn4, \
                _dsn5 = sess.run([image_path_tensor, annotation_tensor,
                                  cost_reduce_mean,
                                  dsn_fuse,
                                  dsn1, dsn2,
                                  dsn3, dsn4,
                                  dsn5],
                                 feed_dict=feed_dict_to_use)
                print("Step: {}, Current Mean Loss: {}".format(step, loss_mean))

                plot_and_save_image(_input_image_path[0], _input_annotation[0],
                                    _dsn_fuse[0], _dsn1[0], _dsn2[0], _dsn3[0], _dsn4[0], _dsn5[0],
                                    args.debug_image_dir, suffix='{}'.format(step))

        ##############
        coord.request_stop()
        coord.join(threads)
        print("Train Finished!")