"""Residual encoder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from model import resnet_v2

ENCODER_DEFUALT_PARAM = {
    "block_name": ["block1", "block2", "block3", "block4"],
    "base_depth": [16, 32, 64, 128],
    "num_units" : [2, 2, 2, 6],
    "stride"    : [2, 1, 1, 1]
}

class EncoderResnet(object):
  """ Residual encoder using off-the-shelf interface.
  """
  def __init__(self, params, mode):
    self.params = params
    self.mode = mode
    self.encoder_params = ENCODER_DEFUALT_PARAM

  def __call__(self, features):
    """ Define tf graph.
    """
    inputs = features['image']

    with tf.variable_scope('encoder') as vsc:
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        # conv1
        with arg_scope(
            [layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
          net = resnet_utils.conv2d_same(inputs, 16, 5, stride=2, scope='conv1')
        tf.add_to_collection(vsc.original_name_scope, net)

        # resnet blocks
        blocks = []
        for i in range(len(self.encoder_params['block_name'])):
          block = resnet_v2.resnet_v2_block(
              scope=self.encoder_params['block_name'][i],
              base_depth=self.encoder_params['base_depth'][i],
              num_units=self.encoder_params['num_units'][i],
              stride=self.encoder_params['stride'][i])
          blocks.append(block)
        net, _ = resnet_v2.resnet_v2(
            net,
            blocks,
            is_training=(self.mode == ModeKeys.TRAIN),
            global_pool=False,
            output_stride=2,
            include_root_block=False,
            scope='resnet')

        tf.add_to_collection(vsc.original_name_scope, net)
    return net
