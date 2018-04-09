""" Hooks used for learn.Experiment
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os
import six
import tensorflow as tf

@six.add_metaclass(abc.ABCMeta)
class Prediction(tf.train.SessionRunHook):
  """ Write predictions to file.
  """
  def __init__(self, params, output_dir):
    self.params = params
    self.output_dir = output_dir

  def begin(self):
    # pylint: disable=attribute-defined-outside-init
    # fetch tensors
    self.predicted_text = tf.get_collection('prediction')[0]["predicted_text"]
    self.image_names = tf.get_collection('prediction')[0]["image_names"]

    # file handle
    file_name = "{}.log".format(self.params['dataset']['dataset_name'])
    file_name = os.path.join(self.output_dir, file_name)
    # log to file
    self.file = open(file_name, 'w')

  def before_run(self, _run_context):
    fetches = {}
    fetches["predicted_text"] = self.predicted_text
    fetches["image_names"] = self.image_names
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    predicted_text_batch = run_values.results["predicted_text"]
    image_name_batch = run_values.results["image_names"]
    assert len(predicted_text_batch) == len(image_name_batch)
    for i in range(len(predicted_text_batch)):
      image_name = image_name_batch[i]
      text = predicted_text_batch[i].decode('utf-8').replace(u'\u2591', '')
      text = text.encode("ascii")
      line = '{}, "{}"'.format(image_name, text)
      tf.logging.info(line)
      self.file.write(line + '\r\n')
    self.file.flush()

  def end(self, _session):
    # disable log to file
    self.file.close()


@six.add_metaclass(abc.ABCMeta)
class FalsePrediction(tf.train.SessionRunHook):
  """ Write false predictions to file.
  """
  def __init__(self, params, output_dir):
    self.params = params
    self.output_dir = output_dir

  def begin(self):
    # pylint: disable=attribute-defined-outside-init
    # fetch tensors
    self.predicted_text = tf.get_collection('prediction')[0]["predicted_text"]
    self.gt_text = tf.get_collection('prediction')[0]["gt_text"]
    self.image_names = tf.get_collection('prediction')[0]["image_names"]
    self.predicted_mask = tf.get_collection('predicted_mask')[0]

    # file handle
    file_name = "{}-false.log".format(self.params['dataset']['dataset_name'])
    file_name = os.path.join(self.output_dir, file_name)
    # log to file
    self.file = open(file_name, 'w')
    self.file.write("image name, ground-truth text, predicted text\r\n")

  def before_run(self, _run_context):
    fetches = {}
    fetches["predicted_text"] = self.predicted_text
    fetches["gt_text"] = self.gt_text
    fetches["image_names"] = self.image_names
    fetches["predicted_mask"] = self.predicted_mask
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    predicted_text = run_values.results["predicted_text"]
    true_text = run_values.results["gt_text"]
    image_names = run_values.results["image_names"]
    predicted_mask = run_values.results["predicted_mask"]
    assert len(predicted_text) == len(image_names)
    assert len(true_text) == len(predicted_mask)
    for i in range(len(predicted_text)):
      if predicted_mask[i]:
        # true prediction
        continue
      image_name = image_names[i]
      pt_text = predicted_text[i].decode('utf-8').replace(u'\u2591', '')
      pt_text = pt_text.encode("ascii")
      gt_text = true_text[i].decode('utf-8').replace(u'\u2591', '')
      gt_text = gt_text.encode("ascii")
      line = '{}, {}, {}\r\n'.format(image_name, gt_text, pt_text)
      self.file.write(line)
    self.file.flush()

  def end(self, _session):
    # disable log to file
    self.file.close()
