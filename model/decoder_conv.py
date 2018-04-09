"""Convoltuional decoder with attention and language ensemble.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import six
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.seq2seq.python.ops.decoder import Decoder
from tensorflow.contrib.seq2seq.python.ops.decoder import _transpose_batch_time
from tensorflow.contrib.seq2seq.python.ops.decoder import dynamic_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers import layer_norm
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from utils import beam_search

DecoderOutput = collections.namedtuple(
    "DecoderOutput",
    ["logits", "predicted_ids", "attention_scores"]
)

BeamDecoderOutput = collections.namedtuple(
    "BeamDecoderOutput",
    ["logits", "predicted_ids", "attention_scores",
     "log_probs", "scores", "beam_parent_ids"]
)

DECODER_DEFUALT_PARAM = {
    "cnn_layers": 6,
    "cnn_hiddens": [512, 512, 512, 512, 512, 512],
    "cnn_kernel": 3,
    "position_embeddings": True,
    "nout_embed": 256,
    "embedding_dim": 512
}

class DecoderConv(object):
  """ Main decoder class.
  """
  def __init__(self, params, mode, num_charset):
    self.params = params
    self.params.update(DECODER_DEFUALT_PARAM)
    self.mode = mode
    self.num_charset = num_charset
    self.max_sequence_length = self.params['dataset']['max_sequence_length']

  def __call__(self, encoder_output, labels):
    if self.mode == ModeKeys.TRAIN:
      with tf.variable_scope("decoder"):
        outputs = self.conv_decoder_train(encoder_output, labels)
        return outputs
    else:
      outputs, _, __ = self.conv_decoder_infer(encoder_output)
      return outputs

  def conv_decoder_train(self, encoder_output, labels):
    label_input = labels['label_input']
    length = labels['length']
    conv_block = ConvBlock(self.params,
                           self.num_charset,
                           is_training=True)

    next_layer = self.add_embedding(label_input, length)

    language, attention, att_scores = conv_block(encoder_output, next_layer)

    language_logit = _transpose_batch_time(language)
    attention_logit = _transpose_batch_time(attention)
    ensemble_logit = language_logit + attention_logit

    sample_ids = tf.cast(tf.argmax(ensemble_logit, axis=-1), tf.int32)

    return DecoderOutput(logits=(language_logit, attention_logit),
                         predicted_ids=sample_ids,
                         attention_scores=att_scores)

  def conv_decoder_infer(self, encoder_output):
    beam_decoder = BeamDecoder(self.params, self.mode,
                               encoder_output, self.num_charset)
    # As tensorflow does not support initializing variable with tensor
    # in a loop or conditional
    beam_decoder.init_params_in_loop()
    tf.get_variable_scope().reuse_variables()

    outputs, final_state, final_sequence_lengths = dynamic_decode(
        decoder=beam_decoder,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=self.max_sequence_length,
        scope='decoder')

    return outputs, final_state, final_sequence_lengths

  def add_embedding(self, labels, length):
    """ Add embeddings for labels
    Args:
      labels: The labels with shape [N, T], where N is batch size
        and T s time steps.
      length: The length for time steps with shape [N]
    Returns:
      The embeded labels
    """
    with tf.variable_scope("embedding"):
      # label embedding
      label_embedding = tf.get_variable(
          name="W",
          shape=[self.num_charset, self.params["embedding_dim"]],
          initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
      next_layer = tf.nn.embedding_lookup(label_embedding, labels)

      # position embedding
      if self.params["position_embeddings"]:
        position_embeding = tf.get_variable(
            name="W_pos",
            shape=[self.max_sequence_length,
                   self.params["embedding_dim"]],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1))

        # Replicate encodings for each element in the batch
        batch_size = tf.shape(length)[0]
        pe_batch = tf.tile([position_embeding], [batch_size, 1, 1])

        # Mask out positions that are padded
        positions_mask = tf.sequence_mask(
            lengths=length, maxlen=self.max_sequence_length, dtype=tf.float32)
        positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

        next_layer = tf.add(next_layer, positions_embed)

      return next_layer

@six.add_metaclass(abc.ABCMeta)
class BeamDecoder(Decoder):
  """Decoder using beam search in eval and infer.
  """
  def __init__(self, params, mode, encoder_outputs, num_charset):
    self.params = params
    self.mode = mode
    self.num_charset = num_charset
    self.max_sequence_length = self.params['dataset']['max_sequence_length']
    self.initial_state = encoder_outputs
    self.fm_height = encoder_outputs.get_shape()[1]
    self.fm_width = encoder_outputs.get_shape()[2]

    # convolution net
    self.conv_block = ConvBlock(self.params,
                                self.num_charset,
                                is_training=False)

    # TODO(Shancheng): use start token and end token rather constant 0
    self.start_token = 0
    self.end_token = 0

    # beam search config
    self.config = beam_search.BeamSearchConfig(
        beam_width=self.params['beam_width'],
        vocab_size=self.num_charset,
        eos_token=self.end_token,
        length_penalty_weight=1.0,
        choose_successors_fn=beam_search.choose_top_k)

  @property
  def batch_size(self):
    return self.params['beam_width']

  @property
  def output_size(self):
    return BeamDecoderOutput(
        logits=tf.TensorShape([self.num_charset]),
        predicted_ids=tf.TensorShape([]),
        attention_scores=tf.TensorShape([self.params['cnn_layers'],
                                         self.fm_height,
                                         self.fm_width]),
        log_probs=tf.TensorShape([]),
        scores=tf.TensorShape([]),
        beam_parent_ids=tf.TensorShape([]))

  @property
  def output_dtype(self):
    return BeamDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        attention_scores=tf.float32,
        log_probs=tf.float32,
        scores=tf.float32,
        beam_parent_ids=tf.int32)

  def initialize(self, name=None):
    finished = tf.tile([False], [self.params['beam_width']])

    start_tokens_batch = tf.fill([self.params['beam_width']], self.start_token)
    first_inputs = self.add_embedding(start_tokens_batch, time=tf.constant(0))

    zeros_padding = tf.zeros([self.params['beam_width'],
                              self.max_sequence_length - 1,
                              first_inputs.get_shape().as_list()[-1]])
    first_inputs = tf.concat([first_inputs, zeros_padding], axis=1)

    beam_state = beam_search.create_initial_beam_state(self.config)

    encoder_output = tf.tile(self.initial_state,
                             [self.params['beam_width'], 1, 1, 1])

    return finished, first_inputs, (encoder_output, beam_state)

  def step(self, time, inputs, state, name=None):
    encoder_output, beam_state = state
    cur_inputs = inputs[:, 0:time + 1, :]
    zeros_padding = inputs[:, time + 2:, :]

    language, attention, scores = self.conv_block(encoder_output,
                                                  cur_inputs)
    # TODO(Shancheng): now it is add operation
    logits = language + attention
    shape = logits.get_shape().as_list()
    logits = tf.reshape(logits, [-1, shape[-1]])

    bs_output, beam_state = beam_search.beam_search_step(
        time_=time,
        logits=logits,
        beam_state=beam_state,
        config=self.config)

    finished, next_inputs = self.next_inputs(bs_output.predicted_ids, (time+1))

    next_inputs = tf.reshape(next_inputs,
                             [self.params['beam_width'], 1,
                              inputs.get_shape().as_list()[-1]])
    next_inputs = tf.concat([cur_inputs, next_inputs], axis=1)
    next_inputs = tf.concat([next_inputs, zeros_padding], axis=1)
    next_inputs.set_shape([self.params['beam_width'],
                           self.max_sequence_length,
                           inputs.get_shape().as_list()[-1]])

    outputs = BeamDecoderOutput(
        logits=logits,
        predicted_ids=bs_output.predicted_ids,
        attention_scores=scores,
        log_probs=beam_state.log_probs,
        scores=bs_output.scores,
        beam_parent_ids=bs_output.beam_parent_ids)
    return outputs, (encoder_output, beam_state), next_inputs, finished

  def finalize(self, output, final_state, sequence_lengths):
    # Gather according to beam search result
    # now predicted_ids is [M, N/B]
    predicted_ids = beam_search.gather_tree(output.predicted_ids,
                                            output.beam_parent_ids)
    # TODO(Shancheng): pay attention
    beam_width = output.beam_parent_ids.get_shape().as_list()
    parent_ids = tf.concat([tf.zeros([1, beam_width[-1]], dtype=tf.int32),
                            output.beam_parent_ids[:-1, :]], 0)
    # now logits is [M, N/B, C]
    logits = beam_search.gather_tree(output.logits,
                                     parent_ids)
    # now attention scores is [M, N/B, L, H, W]
    attention_scores = beam_search.gather_tree(output.attention_scores,
                                               parent_ids)
    # orginal length is the length of ungathered logits
    sequence_lengths = math_ops.not_equal(predicted_ids, self.end_token)
    sequence_lengths = tf.to_int32(sequence_lengths)
    sequence_lengths = tf.reduce_sum(sequence_lengths, axis=0) + 1

    # choose the top score item
    predicted_ids = predicted_ids[:, 0:1]
    logits = logits[:, 0:1]
    attention_scores = attention_scores[:, 0:1]
    # mask out
    length = sequence_lengths[0]
    logits = logits[0:length, :]
    attention_scores = attention_scores[0:length, :]

    final_outputs = DecoderOutput(
        logits=self._padding(logits),
        predicted_ids=self._padding(predicted_ids),
        attention_scores=attention_scores)

    return final_outputs, final_state

  def add_embedding(self, labels, time):
    """ Add embedding in current time step.
    Args:
      labels: The labels with shape [beam_width,] or [beam_width,1]
      time: The time index
    Rreturn:
      The embeded labels
    """
    with tf.variable_scope("embedding"):
      rank = len(labels.shape)
      assert rank == 1 or rank == 2, "labels must be rank 1 or 2"
      if rank == 1:
        labels = tf.expand_dims(labels, axis=1)
      # label embedding
      label_embedding = tf.get_variable(
          name="W",
          shape=[self.num_charset, self.params["embedding_dim"]],
          initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
      next_input = tf.nn.embedding_lookup(label_embedding, labels)

      # position embedding
      if self.params["position_embeddings"]:
        position_embeding = tf.get_variable(
            name="W_pos",
            shape=[self.max_sequence_length,
                   self.params["embedding_dim"]],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1))

        seq_pos_embed = position_embeding[time, :]
        seq_pos_embed = tf.reshape(seq_pos_embed, [1, 1, -1])
        seq_pos_embed_batch = tf.tile(seq_pos_embed,
                                      [self.params['beam_width'], 1, 1])
        next_input = tf.add(next_input, seq_pos_embed_batch)

      return next_input

  def next_inputs(self, sample_ids, time):
    def true_fn():
      # If we're in the last time step
      finished = tf.fill(sample_ids.get_shape().as_list(), True)
      next_inputs = tf.zeros([self.params['beam_width'], 1,
                              self.params["embedding_dim"]])
      return finished, next_inputs

    def false_fn():
      finished = math_ops.equal(sample_ids, self.end_token)
      all_finished = math_ops.reduce_all(finished)
      end_tokens = tf.tile([self.end_token], [self.params['beam_width']])
      next_inputs = control_flow_ops.cond(
          all_finished,
          # If we're finished, the next_inputs value doesn't matter
          lambda: self.add_embedding(end_tokens, time),
          lambda: self.add_embedding(sample_ids, time))
      return finished, next_inputs

    finished = (time >= self.max_sequence_length)
    return control_flow_ops.cond(finished, true_fn, false_fn)

  def init_params_in_loop(self):
    with tf.variable_scope("decoder"):
      _, initial_inputs, initial_state = self.initialize()
      enc_output, _ = initial_state
      # pylint: disable=attribute-defined-outside-init
      self.conv_block.is_init = True
      self.conv_block(enc_output, initial_inputs)
      self.conv_block.is_init = False

  def _padding(self, tensor):
    """ Pad output to max_sequence length,
        for example, paddings = [[0, pad_time],[0,0]]
    """
    shape = tf.shape(tensor)
    pad_time = tf.expand_dims(self.max_sequence_length - shape[0], 0)
    zeros = tf.zeros_like(shape, dtype=shape.dtype)
    paddings = tf.concat([pad_time, zeros[1:]], 0)
    paddings = tf.stack([zeros, paddings], 1)
    return tf.pad(tensor, paddings)

class ConvBlock(object):
  """Basic operation.
  """
  def __init__(self, params, num_charset, is_training=True):
    self.num_charset = num_charset
    self.is_training = is_training
    self.params = params
    self.max_sequence_length = self.params['dataset']['max_sequence_length']

  def __call__(self, encoder_output, input_embed):
    output_collection = tf.get_variable_scope().name
    next_layer = input_embed
    att_scores = []

    # 1D convolution
    for layer_idx in range(self.params['cnn_layers']):
      with tf.variable_scope("conv" + str(layer_idx)):
        nout = self.params['cnn_hiddens'][layer_idx]

        # language module
        # special process here, first padd then conv,
        # because tf does not suport padding other than SAME and VALID
        kernal_width = self.params['cnn_kernel']
        paddings = [[0, 0], [kernal_width - 1, kernal_width - 1], [0, 0]]
        language_layer = tf.pad(next_layer, paddings, "CONSTANT")
        # 1D convolution
        language_layer = self.conv1d_weightnorm(
            inputs=language_layer,
            out_dim=nout * 2,
            kernel_size=kernal_width,
            padding="VALID",
            output_collection=output_collection)
        # to avoid using future information
        language_layer = language_layer[:, 0:-kernal_width + 1, :]

        # add GLU
        language_layer = self.gated_linear_units(language_layer,
                                                 output_collection)

        # shortcut and layer norm
        language_layer = language_layer + next_layer
        language_layer = layer_norm(language_layer,
                                    begin_norm_axis=2,
                                    scope='glu')

        # attention module
        att_out, att_score = self.make_attention(input_embed,
                                                 encoder_output,
                                                 next_layer,
                                                 output_collection)

        # shortcut and layer norm
        attention_layer = att_out + next_layer
        attention_layer = layer_norm(attention_layer,
                                     begin_norm_axis=2,
                                     scope='attention')

        # TODO(Shancheng): now it is add operation
        next_layer = language_layer + attention_layer

        if self.is_training:
          tf.add_to_collection(output_collection, next_layer)
        att_scores.append(att_score)

    # shape=[layer_num, batch_size / beam_width, step, height, width]
    att_scores = tf.stack(att_scores)
    # shape=[batch_size / beam_width, layer_num, step, height, width]
    att_scores = tf.transpose(att_scores, [1, 0, 2, 3, 4])

    language_logit, scores = self.create_logit(language_layer,
                                               att_scores,
                                               output_collection,
                                               "language_logit")
    attention_logit, scores = self.create_logit(attention_layer,
                                                att_scores,
                                                output_collection,
                                                "attention_logit")
    return language_logit, attention_logit, scores

  def create_logit(self, next_layer, att_scores, output_collection, scope):
    # output
    with tf.variable_scope(scope):
      if not self.is_training:
        # only keep the last time step
        # [N/B, M, C] --> [N/B, 1, C]
        next_layer = next_layer[:, -1:, :]
        # [N/B, L, M, H, W] --> [N/B, L, H, W]
        att_scores = att_scores[:, :, -1, :, :]

      next_layer = self.linear_mapping_weightnorm(
          next_layer,
          out_dim=self.params["nout_embed"],
          output_collection=output_collection)
      next_layer = layer_norm(next_layer, begin_norm_axis=2)
      next_layer = self.linear_mapping_weightnorm(
          next_layer,
          out_dim=self.num_charset,
          var_scope_name="liear_logits",
          output_collection=output_collection)

    return next_layer, att_scores

  def linear_mapping_weightnorm(self, inputs, out_dim,
                                var_scope_name="linear",
                                output_collection=None):
    with tf.variable_scope(var_scope_name):
      # pylint: disable=invalid-name
      input_shape = inputs.get_shape().as_list()  # static shape. may has None
      # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
      V = tf.get_variable(
          name='V',
          shape=[int(input_shape[-1]), out_dim],
          dtype=tf.float32,
          initializer=initializers.variance_scaling_initializer())
      # V shape is M*N,  V_norm shape is N
      V_norm = tf.norm(V.initialized_value(), axis=0)
      g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm)
      # weightnorm bias is init zero
      b = tf.get_variable(
          name='b',
          shape=[out_dim],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())

      assert len(input_shape) == 3
      inputs = tf.reshape(inputs, [-1, input_shape[-1]])
      inputs = tf.matmul(inputs, V)
      inputs = tf.reshape(inputs, [input_shape[0], -1, out_dim])

      # g/2-norm(v)
      scaler = tf.div(g, tf.norm(V, axis=0))
      # x*v g/2-norm(v) + b
      inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])

      if self.is_training:
        tf.add_to_collection(output_collection, inputs)
      return inputs

  def conv1d_weightnorm(self, inputs, out_dim, kernel_size, padding="SAME",
                        var_scope_name="conv1d", output_collection=None):
    with tf.variable_scope(var_scope_name):
      # pylint: disable=invalid-name
      in_dim = int(inputs.get_shape()[-1])
      V = tf.get_variable(
          name='V',
          shape=[kernel_size, in_dim, out_dim],
          dtype=tf.float32,
          initializer=initializers.variance_scaling_initializer())
      # V shape is M*N*k,  V_norm shape is k
      V_norm = tf.norm(V.initialized_value(), axis=[0, 1])
      g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm)
      b = tf.get_variable(
          name='b',
          shape=[out_dim],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())

      # use weight normalization (Salimans & Kingma, 2016)
      W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
      inputs = tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding)
      inputs = tf.nn.bias_add(inputs, b)

      if self.is_training:
        tf.add_to_collection(output_collection, inputs)
      return inputs

  def gated_linear_units(self, inputs, output_collection=None):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    inputs = tf.multiply(input_pass, input_gate)

    if self.is_training:
      tf.add_to_collection(output_collection, inputs)
    return inputs

  def make_attention(self, target_embed, encoder_output,
                     decoder_hidden, output_collection):
    with tf.variable_scope("attention"):
      embed_size = target_embed.get_shape().as_list()[-1]
      hidden_size = decoder_hidden.get_shape().as_list()[-1]

      decoder_rep = decoder_hidden + target_embed
      # character project to image
      decoder_rep = self.linear_mapping_weightnorm(
          decoder_rep,
          out_dim=embed_size,
          var_scope_name="linear_query",
          output_collection=output_collection)

      att_out, att_score = self.attention_score_pooling(decoder_rep,
                                                        encoder_output)
      # image project to character
      att_out = self.linear_mapping_weightnorm(
          att_out,
          out_dim=hidden_size,
          var_scope_name="linear_out",
          output_collection=output_collection)

    return att_out, att_score

  def attention_score_pooling(self, dec_rep, encoder_output):
    # pylint: disable=invalid-name
    # static shape
    N, H, W, C = encoder_output.get_shape().as_list()
    # static shape in train, dynamic shape in infer
    N = N or tf.shape(dec_rep)[0]
    M = dec_rep.get_shape().as_list()[1] or tf.shape(dec_rep)[1]

    encoder_reshape = tf.reshape(encoder_output, [N, H * W, C])  # N*(H*W)*C

    # N*M*C  ** N*(H*W)*C  --> N*M*(H*W)
    att_score = tf.matmul(dec_rep, encoder_reshape,
                          transpose_b=True) * tf.sqrt(1.0 / C)

    att_score = tf.transpose(att_score, [0, 2, 1])  # N*(H*W)*M
    att_score = tf.reshape(att_score, [N, H, W, M])  # N*H*W*M
    att_score = tf.pad(att_score,
                       [[0, 0], [1, 1], [1, 1], [0, 0]],
                       "SYMMETRIC")
    att_score = tf.nn.avg_pool(att_score,
                               [1, 3, 3, 1],
                               [1, 1, 1, 1],
                               padding='VALID')  # N*H*W*M
    att_score = tf.reshape(att_score, [N, H * W, M])  # N*(H*W)*M
    att_score = tf.transpose(att_score, [0, 2, 1])  # N*M*(H*W)
    att_score = tf.nn.softmax(att_score)  # N*M*(H*W)

    # N*M*(H*W) ** N*(H*W)*C  --> N*M*C
    att_out = tf.matmul(att_score, encoder_reshape)
    att_score = tf.reshape(att_score, [N, M, H, W])
    return att_out, att_score

  def attention_score(self, dec_rep, encoder_output):
    # pylint: disable=invalid-name
    # static shape
    N, H, W, C = encoder_output.get_shape().as_list()
    # static shape in train, dynamic shape in infer
    N = N or tf.shape(dec_rep)[0]
    M = dec_rep.get_shape().as_list()[1] or tf.shape(dec_rep)[1]

    encoder_reshape = tf.reshape(encoder_output, [N, H * W, C])  # N*(H*W)*C

    # N*M*C  ** N*(H*W)*C  --> N*M*(H*W)
    att_score = tf.matmul(dec_rep, encoder_reshape,
                          transpose_b=True) * tf.sqrt(1.0 / C)

    att_score = tf.nn.softmax(att_score)  # N*M*(H*W)

    # N*M*(H*W) ** N*(H*W)*C  --> N*M*C
    att_out = tf.matmul(att_score, encoder_reshape)
    att_score = tf.reshape(att_score, [N, M, H, W])
    return att_out, att_score
