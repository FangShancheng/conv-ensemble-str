"""Quality metrics for the model."""

import tensorflow as tf

def char_accuracy(predictions, labels, rej_char=0, streaming=True,
                  ignore_case=True):
  """ Evaluate in character level.
  """
  with tf.variable_scope('CharAccuracy'):
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    targets = tf.to_int32(labels)
    if ignore_case:
      predictions = _lower_to_captical_case(predictions)
      targets = _lower_to_captical_case(targets)
    const_rej_char = tf.fill(tf.shape(targets), rej_char)
    weights = tf.to_float(tf.not_equal(targets, const_rej_char))
    correct_chars = tf.to_float(tf.equal(predictions, targets))
    accuracy_per_example = tf.div(
        tf.reduce_sum(tf.multiply(correct_chars, weights), 1),
        tf.reduce_sum(weights, 1))
    if streaming:
      streaming_mean = tf.contrib.metrics.streaming_mean(accuracy_per_example)
      return streaming_mean
    else:
      return tf.reduce_mean(accuracy_per_example)

def sequence_accuracy(predictions, labels, streaming=True, ignore_case=True):
  """ Evaluate in word level.
  """
  with tf.variable_scope('SequenceAccuracy'):
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    targets = tf.to_int32(labels)
    include_predictions = predictions

    if ignore_case:
      include_predictions = _lower_to_captical_case(include_predictions)
      targets = _lower_to_captical_case(targets)

    correct_chars = tf.to_float(tf.equal(include_predictions, targets))
    correct_chars_counts = tf.cast(
        tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
    target_length = targets.get_shape().dims[1].value
    target_chars_counts = tf.fill(tf.shape(correct_chars_counts), target_length)
    accuracy_per_example = tf.to_float(
        tf.equal(correct_chars_counts, target_chars_counts))
    if streaming:
      streaming_mean = tf.contrib.metrics.streaming_mean(accuracy_per_example)
      tf.add_to_collection('predicted_mask', accuracy_per_example)
      return streaming_mean
    else:
      return tf.reduce_mean(accuracy_per_example)

def _lower_to_captical_case(src):
  # ranks of src can be any
  low, high = 11, 62
  space = (high - low + 1) / 2
  mid_tf = tf.fill(tf.shape(src), high - space + 1)
  high_tf = tf.fill(tf.shape(src), high)
  mid_mask = tf.greater_equal(src, mid_tf)
  high_mask = tf.less_equal(src, high_tf)
  case_mask = tf.logical_and(mid_mask, high_mask)
  return tf.where(case_mask, src - space, src)
