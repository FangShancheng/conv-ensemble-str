#!/usr/bin/env python
# Copyright 2017-2018 IIE, CAS.
# Written by Shancheng Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" A quick demo to recognize text.
"""
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.contrib.learn import ModeKeys
from model.model import Model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, required=True,
                      help='path to image file.')
  parser.add_argument('--checkpoint', type=str, default='data/model.ckpt',
                      help='path to image file.')
  args = parser.parse_args()

  params = {
    'checkpoint': args.checkpoint,
    'dataset':{
      'dataset_dir': 'data',
      'charset_filename': 'charset_size=63.txt',
      'max_sequence_length': 30,
    },
    'beam_width': 1,
    'summary': False
  }
  model = Model(params, ModeKeys.INFER)
  image = tf.placeholder(tf.uint8, (1, 32, 100, 3), name='image')
  predictions, _, _ = model({'image': image}, None)

  assert os.path.exists(args.path), '%s does not exists!' % args.path
  raw_image = Image.open(args.path).convert('RGB')
  raw_image = raw_image.resize((100, 32))
  raw_image = np.array(raw_image)[None, :]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    predictions = sess.run(predictions, feed_dict={image: raw_image})
    text = predictions['predicted_text'][0]
    print('%s: %s' % (args.path, text))

if __name__ == '__main__':
  main()
