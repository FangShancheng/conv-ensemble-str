# Copyright 2017 IIE, CAS.
# Written by Shancheng Fang
# ==============================================================================
"""Define base dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import abc
import sys
import copy
import six

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

def create_dataset(def_dict, mode, use_beam_search):
  """Creates an Dataset object from a dictionary definition.

  Args:
    def_dict: A dictionary defining the input pipeline.
      It must have "dataset_name", "split_name" and "dataset_dir" that
      correspond to the class       name and constructor parameters of
      an InputPipeline, respectively.
    mode: A value in tf.contrib.learn.ModeKeys
    use_beam_search: Whether to use beam search

  Returns:
    A Dataset object.
  """
  if not "dataset_name" in def_dict:
    raise ValueError("Dataset definition must have a dataset_name property.")

  class_ = def_dict["dataset_name"]
  if not hasattr(sys.modules[__name__], class_):
    raise ValueError("Invalid Dataset class: {}".format(class_))

  # TODO(Shancheng): to support batch_size > 1,
  # remove use_beam_search argument
  if mode != tf.contrib.learn.ModeKeys.TRAIN and use_beam_search:
    def_dict['batch_size'] = 1

  dataset_class = getattr(sys.modules[__name__], class_)
  return dataset_class(params=def_dict, mode=mode)

@six.add_metaclass(abc.ABCMeta)
class Dataset():
  """An abstract Dataset class. All datasets must inherit from this.
    This class defines how data is read, parsed, and separated into
    features and labels.
  """
  def __init__(self, params, mode):
    self.mode = mode
    default_params = self._dataset_params()
    self.params = self._parse_params(params, default_params)

  @property
  def _params_template(self):
    """Params placeholder.
    """
    return {
        'dataset_name': None,
        'dataset_dir': None,
        'batch_size': None,
        'splits': None,
        'charset_filename': None,
        'image_shape': None,
        'max_sequence_length': None,
        'null_code': None,
        'shuffle': None,
        'smaller_final_batch': None,
        'num_epochs': None,
        'split_name': None
    }


  def _dataset_params(self):
    """A abstract function implemented by subclass.
    """
    raise NotImplementedError("Not implemented.")

  def _parse_params(self, params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
      params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
      # If param is unknown, drop it to stay compatible with past versions
      if key not in default_params:
        raise ValueError("%s is not a valid model parameter" % key)
      # Param is a dictionary
      if isinstance(value, dict):
        default_dict = default_params[key]
        if not isinstance(default_dict, dict):
          raise ValueError("%s should not be a dictionary", key)
        if default_dict:
          value = self._parse_params(value, default_dict)
        else:
          # If the default is an empty dict we do not typecheck it
          # and assume it's done downstream
          pass
      if value is None:
        continue
      if default_params[key] is None:
        result[key] = value
      else:
        result[key] = type(default_params[key])(value)
    return result

  def _read_from_data_provider(self, data_provider):
    """Utility function to read all available items from a DataProvider.
    """
    list_items = set(data_provider.list_items())
    assert self.items.issubset(list_items), \
           "items are unavailable in data_provider!"

    item_values = data_provider.get(list(self.items))
    items_dict = dict(zip(self.items, item_values))
    return items_dict

  def _make_data_provider(self, **kwargs):
    """Create data provider
    """
    split_name = self.params['split_name']
    if split_name not in self.params['splits']:
      raise ValueError('split name %s was not recognized.' % split_name)

    decoder = tfexample_decoder.TFExampleDecoder(self.keys_to_features,
                                                 self.items_to_handlers)

    file_pattern = os.path.join(self.params['dataset_dir'],
                                self.params['splits'][split_name])

    tf.logging.info("Create dataset.")
    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  def create_input_fn(self):
    """Creates an input function that can be used with tf.learn estimators.
      Note that you must pass "factory funcitons" for both the data provider and
      featurizer to ensure that everything will be created in  the same graph.
    """
    with tf.variable_scope("input_fn"):
      batch_size = self.params['batch_size']
      data_provider = self._make_data_provider()
      features_and_labels = self._read_from_data_provider(data_provider)

      tf.logging.info("Start batch queue.")
      batch = tf.train.batch(
          tensors=features_and_labels,
          enqueue_many=False,
          batch_size=batch_size,
          dynamic_pad=True,
          capacity=3000 + 16 * batch_size,
          allow_smaller_final_batch=self.params['smaller_final_batch'],
          name="batch_queue",
          num_threads=int((batch_size+1)/2)
      )

      # Separate features and labels
      features_batch = {k: batch[k] for k in self.feature_keys}
      if set(batch.keys()).intersection(self.label_keys):
        labels_batch = {k: batch[k] for k in self.label_keys}
      else:
        labels_batch = None

      return features_batch, labels_batch

  @property
  def keys_to_features(self):
    """Key to features
    """
    default = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/text_length':
            tf.FixedLenFeature([1], tf.int64,
                               default_value=tf.zeros([1], dtype=tf.int64)),
        'image/class':
            tf.FixedLenFeature([self.params['max_sequence_length']], tf.int64),
        'image/text':
            tf.FixedLenFeature([1], tf.string, default_value=''),
        'image/name':
            tf.FixedLenFeature((), tf.string, default_value='')
    }
    return default

  @property
  def items_to_handlers(self):
    """Items to handlers
    """
    default = {
        'image': tfexample_decoder.Image(
            shape=self.params['image_shape'],
            image_key='image/encoded',
            format_key='image/format'),
        'label': tfexample_decoder.Tensor(tensor_key='image/class'),
        'text': tfexample_decoder.Tensor(tensor_key='image/text'),
        'length': tfexample_decoder.Tensor(tensor_key='image/text_length'),
        'name': tfexample_decoder.Tensor(tensor_key='image/name')
    }
    return default

  @property
  def items(self):
    """items
    """
    return self.feature_keys.union(self.label_keys)

  @property
  def feature_keys(self):
    """Only image and name supported.
    """
    return set(["image", "name"])

  @property
  def label_keys(self):
    """Only label and length supported.
    """
    return set(["label", "length"])

class MJSynth(Dataset):
  """Training dataset.
  """
  def _dataset_params(self):
    dataset_params = {
        'dataset_name': 'MJSynth',
        'dataset_dir': '/opt/fsc/tf-mjsynth',
        'batch_size': 64,
        'splits': {
            'train': 'train/train-*/*.tfrecord',
            'test': 'test/test-*/*.tfrecord',
            'validation': 'val/val-*/*.tfrecord'
        },
        'charset_filename': 'charset_size=63.txt',
        'image_shape': (32, 100, 3),
        'max_sequence_length': 30,
        'null_code': 0,
    }
    default_params = self._params_template
    default_params.update(dataset_params)
    return default_params
