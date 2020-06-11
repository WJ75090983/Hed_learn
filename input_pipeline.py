#input_pipeline.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from const import const_image_height
from const import const_image_width
from const import const_use_batch_norm
from const import const_use_kernel_regularizer

import tensorflow as tf

##### 固定尺寸的image,不需要tf.image.resize,而是用tf.reshape

def read_fix_size_image_format(dataset_root_dir_string, filename_queue):
  #从csv中加载源图像和标签图像

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  #default values
  record_defaults = [[''],['']]
  image_path, annotation_path = tf.decode_csv(value, record_defaults=record_defaults)

  #csv里保存的不是绝对路径，需要和dataset_root_dir_string一起拼装成完整的路径
  image_path = tf.string_join([tf.constant(dataset_root_dir_string), image_path])
  annotation_path = tf.string_join([tf.constant(dataset_root_dir_string), annotation_path])

  image_content = tf.read_file(image_path)
  annotation_content = tf.read_file(annotation_path)

  #image is jpg, annotation is png
  image_tensor = tf.image.decode_jpeg(image_content, channels=3)
  annotation_tensor = tf.image.decode_png(annotation_content, channels=1)

  #decode之后，一定要设置image的大小，或者resize到一个size，否则会crash
  image_tensor = tf.reshape(image_tensor, [const_image_height, const_image_width, 3])
  annotation_tensor = tf.reshape(annotation_tensor, [const_image_height, const_image_width, 1])

  image_float = tf.to_float(image_tensor)
  annotation_float = tf.to_float(annotation_tensor)

  if const_use_batch_norm == True:
    image_float = image_float/255.0
  else:
    #这个分支主要是为了匹配不使用batch norm时的VGG
    image_float = mean_image_subtraction(image_float, [R_MEAN, G_MEAN, B_MEAN])

  #不管是不是VGG, annotation都需要归一化
  annotation_float = annotation_float / 255.0

  return image_float, annotation_float, image_path

def fix_size_image_pipeline(dataset_root_dir_string, filename_queue, batch_size, num_epochs=None):
  image_tensor, annotation_tensor, image_path = read_fix_size_image_format(dataset_root_dir_string, filename_queue)
  min_after_dequeue = 200
  capacity = min_after_dequeue + 3 * batch_size

  image_tensor, annotation_tensor, image_path_tensor = tf.train.shuffle_batch([image_tensor, annotation_tensor, image_path],
                                          batch_size = batch_size,
                                          capacity = capacity,
                                          min_after_dequeue = min_after_dequeue)
  return image_tensor, annotation_tensor, image_path_tensor