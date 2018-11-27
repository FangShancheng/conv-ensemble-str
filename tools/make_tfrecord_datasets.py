#!/usr/bin/env python
""" A tool to convert mjsynth datasets to tfrecords datasets.
"""
import os
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

SPLITS = ['train', 'test', 'val']
ORIGINAL_DIR = '/home/data/Dataset/mjsynth'
OUTPUT_DIR = '/home/data/Dataset/tf-data/tf-mjsynth'
CHARSET = 'data/charset_size=63.txt'
MAX_WORD_LENGTH = 30
NULL_CODE = 0
NORM_SIZE = (100, 32)
FILE_NUM_LIMIT = 3000

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _get_charset():
  charset_path = os.path.join(ORIGINAL_DIR, CHARSET)
  assert tf.gfile.Exists(charset_path), charset_path
  charset = {}
  with open(charset_path) as file:
    for line in file:
      value, key = line.split()
      charset[key] = value
  return charset

def _get_label(text, charset):
  unpadded_label = []
  for c in text:
    label = int(charset[c])
    unpadded_label.append(label)
  label = [unpadded_label[i] if i < len(unpadded_label) else NULL_CODE for i in range(MAX_WORD_LENGTH)]
  return label, unpadded_label

def _get_text(path):
  return path.split('_')[1].strip()

def run(split, charset):
  annotation = os.path.join(ORIGINAL_DIR, 'annotation_%s.txt'%(split))
  assert tf.gfile.Exists(annotation), annotation
  split_dir = os.path.join(OUTPUT_DIR, split)
  tf.gfile.MkDir(split_dir)

  with tf.Graph().as_default():
    with tf.Session('') as sess:
      image_placeholder = tf.placeholder(tf.uint8)
      encoded_image = tf.image.encode_png(image_placeholder)
      with open(annotation, 'r') as file:
        lines = file.readlines()
      with open('/tmp/fsc-error.log', 'a') as efile:
        fold_index = 0
        for index, line in enumerate(lines):
          image_name, word_label = line.split()
          split_name = os.path.splitext(os.path.split(image_name)[1])[0]
          text = _get_text(image_name)
          text_length = len(text)
          label, unpadded_label = _get_label(text, charset)
          image_path = os.path.join(ORIGINAL_DIR, image_name)
          try:
            image = plt.imread(image_path)
            image = cv2.resize(image, NORM_SIZE)
            width, height = image.shape[0:2]

            jpg_image = sess.run(encoded_image, feed_dict={image_placeholder:image})

            example = tf.train.Example(features=tf.train.Features(feature={
                          'image/encoded': _bytes_feature([jpg_image]),
                          'image/format': _bytes_feature(["PNG"]),
                          'image/class': _int64_feature(label),
                          'image/unpadded_class': _int64_feature(unpadded_label),
                          'image/text': _bytes_feature([text]),
                          'image/text_length': _int64_feature([text_length]),
                          'image/height': _int64_feature([height]),
                          'image/width': _int64_feature([width]),
                          'image/word_class': _int64_feature([int(word_label)]),
                          'image/name':_bytes_feature([split_name])}))
            if index % FILE_NUM_LIMIT == 0:
              fold_name = '%s-%04d'%(split, fold_index)
              fold_dir = os.path.join(split_dir, fold_name)
              tf.gfile.MkDir(fold_dir)
              fold_index += 1

            tfrecords_filename = '%s.tfrecord'%(os.path.splitext(os.path.basename(image_name))[0])
            tfrecords_filename = os.path.join(fold_dir, tfrecords_filename)
            print '[%.2f%%] Writing %s'%(100.0*index/len(lines), tfrecords_filename)
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            writer.write(example.SerializeToString())
          except IOError:
            message = 'bad image: %s\n'%image_path
            efile.write(message)
            efile.flush()
          except Exception as e:
            efile.write('something error:%s\n'%(e.message))
            efile.flush()

def main():
  assert tf.gfile.Exists(ORIGINAL_DIR)
  if tf.gfile.Exists(OUTPUT_DIR):
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  tf.gfile.MkDir(OUTPUT_DIR)
  
  tf.gfile.Copy(os.path.join(ORIGINAL_DIR, CHARSET),
               os.path.join(OUTPUT_DIR, CHARSET))

  charset = _get_charset()
  for split in SPLITS:
    run(split, charset)

if __name__ == '__main__':
  main()
