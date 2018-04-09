""" Top view model interface.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.framework import init_from_checkpoint
from tensorflow.contrib.framework.python.framework import list_variables
from tensorflow.contrib.framework.python.ops import get_variables_to_restore
from tensorflow.contrib.seq2seq.python.ops.decoder import _transpose_batch_time
from model.encoder_resnet import EncoderResnet
from model.decoder_conv import DecoderConv
from utils.utils import augment_image
from utils.utils import CharsetMapper

class Model(object):
  """ Top view model.
  """
  def __init__(self, params, mode):
    # params
    self.params = params
    self.mode = mode

    # charset
    charset_file = os.path.join(self.params['dataset']['dataset_dir'],
                                self.params['dataset']['charset_filename'])
    self.charset = CharsetMapper(charset_file,
                                 self.params['dataset']['max_sequence_length'])

    # endcoder and decoder
    self.encoder = EncoderResnet(params, mode)
    self.decoder = DecoderConv(params, mode, self.charset.num_charset)

    tf.logging.info("Model params in mode=%s", self.mode)
    tf.logging.info("\n%s", yaml.dump({"Model": self.params}))

  def __call__(self, features, labels):
    with tf.variable_scope('model'):
      # Pre-process features and labels
      tf.logging.info('Preprocess data.')
      features, labels = self._preprocess(features, labels)
      tf.logging.info('Create encoder.')
      encoder_output = self.encoder(features)
      tf.logging.info('Create decoder.')
      decoder_output = self.decoder(encoder_output, labels)

      if self.params['checkpoint']:
        self._restore_variables(self.params['checkpoint'])

      # loss is zero during eval
      loss = tf.zeros([])
      train_op = None
      if self.mode == ModeKeys.TRAIN:
        tf.logging.info('Compute loss.')
        loss = self._compute_loss(decoder_output, labels)
        # TODO(Shancheng): gradient multipliers
        train_op = self._build_train_op(loss)
        tf.logging.info('Compute Statistics.')
        self._compute_statistics()

      if self.params['summary']:
        tf.logging.info('Create summaries.')
        self._create_summaries(decoder_output, features, labels)

      tf.logging.info('Create predictions.')
      predictions = self._create_predictions(decoder_output, features, labels)
      tf.logging.info('Model done.')
      return predictions, loss, train_op

  def _restore_variables(self, checkpoint):
    """ restore variables from checkpoint as much as possible
    """
    checkpoint_variables_map = list_variables(checkpoint)
    valid_variable = lambda name: name.startswith('model/encoder') or \
                                  name.startswith('model/decoder')
    checkpoint_variable_names = [name for (name, _) in checkpoint_variables_map
                                 if valid_variable(name)]

    variables = get_variables_to_restore()
    variable_names = [v.name.split(':')[0] for v in variables]
    assignment_map = {}
    for var in checkpoint_variable_names:
      if var in variable_names:
        assignment_map[var] = var

    init_from_checkpoint(checkpoint, assignment_map)

  def _preprocess(self, features, labels):
    """ Data augmentation and label process.
    """
    with tf.variable_scope('preprocess'):
      with tf.variable_scope('image'):
        features['image_orig'] = features['image']
        image = tf.image.convert_image_dtype(features['image_orig'],
                                             dtype=tf.float32)
        if self.mode == ModeKeys.TRAIN:
          images = tf.unstack(image)
          images = [augment_image(img) for img in images]
          image = tf.stack(images)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        features['image'] = image

      if labels is None:
        return features, None

      with tf.variable_scope('label'):
        # TODO(Shancheng): use start token and end token rather constant 0
        # labels for decoder input
        labels['label_input'] = tf.concat([labels['label'][:, -1:],
                                           labels['label'][:, 0:-1]], axis=1)
        # from text length to training label length
        labels['length'] = tf.reshape(labels['length'], [-1])
        labels['length'] = labels['length'] + 1

    return features, labels

  def _build_train_op(self, loss, gradient_multipliers=None):
    """Creates the training operation"""
    # Creates the optimizer
    name = self.params["optimizer"]
    optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
        learning_rate=self.params["learning_rate"],
        **self.params["optimizer_params"])

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=self.params["learning_rate"],
        learning_rate_decay_fn=None,
        clip_gradients=self.params['clip_gradients'],
        optimizer=optimizer,
        gradient_multipliers=gradient_multipliers,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return train_op

  def _create_predictions(self, decoder_output, features, labels=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
    with tf.name_scope("create_predictions"):
      predicted_ids = _transpose_batch_time(decoder_output.predicted_ids)
      predicted_text = self.charset.get_text(predicted_ids)
      attention_scores = decoder_output.attention_scores
      original_images = features["image_orig"]
      prediction = {"predicted_ids": predicted_ids,
                    "predicted_text": predicted_text,
                    "images": original_images,
                    "attention_scores": attention_scores}
      if "name" in features:
        prediction["image_names"] = features['name']
      if labels:
        gt_text = self.charset.get_text(labels["label"])
        prediction["gt_text"] = gt_text
      tf.add_to_collection("prediction", prediction)
      return prediction

  def _create_summaries(self, decoder_output, features, labels=None):
    """Create summaries for tensorboard.
    """
    with tf.name_scope("create_summaries"):
      max_outputs = self.params['max_outputs']

      # input images
      image = features['image']
      tf.summary.image(self._sname('image'), image, max_outputs)
      if self.mode == ModeKeys.TRAIN:
        image_orig = features['image_orig']
        tf.summary.image(self._sname('image_orig'), image_orig, max_outputs)

      # ground-truth text
      if self.mode != ModeKeys.INFER:
        gt_text = self.charset.get_text(labels["label"][:max_outputs, :])
        tf.summary.text(self._sname('text/gt'), gt_text)

      # predicted text
      predicted_ids = _transpose_batch_time(decoder_output.predicted_ids)
      predicted_ids = tf.to_int64(predicted_ids[:max_outputs, :])
      predicted_text = self.charset.get_text(predicted_ids)
      tf.summary.text(self._sname('text/pt'), predicted_text)

      def add_attention_summary(att_scores, family='attention'):
        for att_score in att_scores:
          name = att_score.name.replace(":", "_")
          shape = tf.shape(att_score)
          # pylint: disable=invalid-name
          N, M, H, W = shape[0], shape[1], shape[2], shape[3]
          score = tf.reshape(att_score, [N, M * H, W])
          score = tf.expand_dims(score, 3)
          tf.summary.image(name, score, max_outputs=max_outputs, family=family)

      def add_std_max_summary(tensors, family):
        for tensor in tensors:
          name = tensor.name.replace(":", "_")
          _, var = tf.nn.moments(tf.reshape(tensor, [-1]), [0])
          tf.summary.scalar(name, tf.sqrt(var), family=family + "_std")
          max_value = tf.reduce_max(tensor)
          tf.summary.scalar(name, max_value, family=family + "_max")

      # attention scores [N, L, M, H, W]
      attention_scores = decoder_output.attention_scores
      # unstack layer
      attention_scores = tf.unstack(attention_scores, axis=1)
      add_attention_summary(attention_scores, 'attention')

      # weight
      weigths = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      add_std_max_summary(weigths, 'weights')

      # conv1 and encoder output activation
      encoder_outputs = tf.get_collection('model/encoder/')
      add_std_max_summary(encoder_outputs, 'activation')

      # encoder activation
      encoder_outputs = tf.get_collection('model/encoder/resnet/_end_points')
      add_std_max_summary(encoder_outputs, 'activation')

      # decoder activation
      decoder_outputs = tf.get_collection('model/decoder')
      add_std_max_summary(decoder_outputs, 'activation')

  # TODO(Shancheng): use tensorflow loss interface
  def _compute_loss(self, decoder_output, labels):
    """Computes the loss for this model.
    """
    with tf.name_scope("compute_loss"):
      language_logit = decoder_output.logits[0]
      attention_logit = decoder_output.logits[1]
      batch_size = self.params['dataset']['batch_size']

      language_losses = self._cross_entropy_sequence_loss(
          logits=language_logit,
          targets=tf.transpose(labels["label"], [1, 0]),
          sequence_length=labels["length"])
      attention_losses = self._cross_entropy_sequence_loss(
          logits=attention_logit,
          targets=tf.transpose(labels["label"], [1, 0]),
          sequence_length=labels["length"])

      language_loss = tf.reduce_sum(language_losses) / batch_size
      attention_loss = tf.reduce_sum(attention_losses) / batch_size
      loss = language_loss + attention_loss

      return loss

  def _cross_entropy_sequence_loss(self, logits, targets, sequence_length):
    """Calculates the per-example cross-entropy loss for a sequence of logits
      and masks out all losses passed the sequence length.

    Args:
      logits: Logits of shape `[T, B, vocab_size]`
      targets: Target classes of shape `[T, B]`
      sequence_length: An int32 tensor of shape `[B]` corresponding
        to the length of each input

    Returns:
      A tensor of shape [T, B] that contains the loss per example,
      per time step.
    """
    with tf.name_scope("cross_entropy_sequence_loss"):
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=targets)

      # Mask out the losses we don't care about
      loss_mask = tf.sequence_mask(
          tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
      losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

      return losses

  def _compute_statistics(self):
    """ Compute parameter number and flops.
    """
    # log to file
    output_dir = self.params['output_dir']
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'statistics.log')
    log = logging.getLogger('tensorflow')
    handle = logging.FileHandler(output_dir)
    log.addHandler(handle)

    # FLOPS
    encoder_flops, decoder_flops = 0, 0
    encoder_count, decoder_count = 0, 0
    graph = tf.get_default_graph()
    for operation in graph.get_operations():
      flops = ops.get_stats_for_node_def(graph, operation.node_def,
                                         'flops').value
      if flops is None:
        continue
      if operation.name.startswith('model/encoder'):
        # encoder
        encoder_flops += flops
        encoder_count += 1
        tf.logging.info('encoder operation %s : %d', operation.name, flops)
      elif operation.name.startswith('model/decoder'):
        # decoder
        decoder_flops += flops
        decoder_count += 1
        tf.logging.info('decoder operation %s : %d', operation.name, flops)
      else:
        # gradient
        pass
    tf.logging.info('flops of %d encoder tensor: %d',
                    encoder_count, encoder_flops)
    tf.logging.info('flops of %d decoder tensor: %d',
                    decoder_count, decoder_flops)
    tf.logging.info('flops of total %d tensor: %d',
                    encoder_count + decoder_count,
                    encoder_flops + decoder_flops)
    # parameters
    encoder_parameters, decoder_parameters = 0, 0
    encoder_count, decoder_count = 0, 0
    for var in tf.trainable_variables():
      parameters = np.prod(var.get_shape().as_list())
      if var.name.startswith('model/encoder'):
        # encoder
        encoder_parameters += parameters
        encoder_count += 1
        tf.logging.info('encoder variable %s : %d', var.name, parameters)
      elif var.name.startswith('model/decoder'):
        # decoder
        decoder_parameters += parameters
        decoder_count += 1
        tf.logging.info('decoder variable %s : %d', var.name, parameters)

    tf.logging.info('parameters of %d encoder tensor: %d',
                    encoder_count, encoder_parameters)
    tf.logging.info('parameters of %d decoder tensor: %d',
                    decoder_count, decoder_parameters)
    tf.logging.info('parameters of total %d tensor: %d',
                    encoder_count + decoder_count,
                    encoder_parameters + decoder_parameters)
    # disable log to file
    log.removeHandler(handle)

  def _sname(self, label):
    """ Utility.
    """
    if self.mode == ModeKeys.TRAIN:
      prefix = 'train'
    elif self.mode == ModeKeys.EVAL:
      prefix = 'eval'
    else:
      prefix = 'infer'
    return '%s/%s' % (prefix, label)
