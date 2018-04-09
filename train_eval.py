#!/usr/bin/env python
# Copyright 2017 IIE, CAS.
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

"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config

import config
import datasets
from model.model import Model
from utils.hooks import Prediction, FalsePrediction
from utils.metrics import sequence_accuracy, char_accuracy

FLAGS = tf.flags.FLAGS
config.define()

def _create_dataset_params():
  """Create dataset params
  """
  dparams = {
      "dataset_name": FLAGS.dataset_name,
      "dataset_dir": FLAGS.dataset_dir,
      "batch_size": FLAGS.batch_size
  }

  if FLAGS.schedule == 'train':
    split_name = FLAGS.split_name or 'train'
    dparams.update({
        'shuffle': True,
        'smaller_final_batch': False,
        'num_epochs': None,
        'split_name': split_name})
  elif FLAGS.schedule == 'evaluate':
    split_name = FLAGS.split_name or 'test'
    dparams.update({
        'shuffle': False,
        'smaller_final_batch': True,
        'num_epochs': 1,
        'split_name': split_name})
  else:
    split_name = FLAGS.split_name or 'test'
    dparams.update({
        'shuffle': False,
        'smaller_final_batch': False,
        'num_epochs': None,
        'split_name': split_name})
  return dparams

def _create_model_params(dataset):
  """Create model params
  """
  mparams = {
      "optimizer": FLAGS.optimizer,
      "learning_rate": FLAGS.learning_rate,
      "clip_gradients": FLAGS.clip_gradients,
      "dataset": dataset.params,
      "optimizer_params": {
          "momentum": FLAGS.momentum,
          "use_nesterov": FLAGS.use_nesterov
      },
      "summary": FLAGS.summary,
      "max_outputs": FLAGS.max_outputs,
      "beam_width": FLAGS.beam_width,
      "output_dir": FLAGS.output_dir,
      "checkpoint": FLAGS.checkpoint
  }
  return mparams

def _create_hooks(mparams, output_dir):
  """Create hooks
  """
  # Create training hooks
  train_hooks = []
  # Create evaluating hooks and eval config
  eval_hooks = []

  # Write prediction to file
  prediction_hook = Prediction(mparams, FLAGS.output_dir)
  eval_hooks.append(prediction_hook)

  # Write false prediction to file
  false_prediction_hook = FalsePrediction(mparams, FLAGS.output_dir)
  eval_hooks.append(false_prediction_hook)

  if FLAGS.schedule == 'continuous_eval':
    eval_output_dir = os.path.join(output_dir, 'eval_continuous')
    eval_hooks.append(tf.contrib.training.SummaryAtEndHook(eval_output_dir))
  elif FLAGS.schedule == 'evaluate':
    # stop until data are exhausted
    FLAGS.eval_steps = None

  if FLAGS.debug:
    from tensorflow.python import debug as tf_debug
    debug_hook = tf_debug.LocalCLIDebugHook()
    train_hooks.append(debug_hook)
    eval_hooks.append(debug_hook)
  return train_hooks, eval_hooks

def _create_experiment(output_dir):
  """
  Creates a new Experiment instance.

  Args:
    output_dir: Output directory for model checkpoints and summaries.
  """
  # Runconfig
  session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(
      per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction,
      allow_growth=FLAGS.gpu_allow_growth))
  estimator_config = run_config.RunConfig(
      session_config=session_config,
      gpu_memory_fraction=FLAGS.gpu_memory_fraction,
      tf_random_seed=FLAGS.tf_random_seed,
      log_step_count_steps=FLAGS.log_step,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

  # Dataset
  mode = tf.contrib.learn.ModeKeys.TRAIN if FLAGS.schedule == 'train' \
                                    else tf.contrib.learn.ModeKeys.EVAL
  dataset = datasets.create_dataset(
      def_dict=_create_dataset_params(),
      mode=mode,
      use_beam_search=FLAGS.beam_width)

  # Model function
  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    model = Model(params, mode)
    predictions, loss, train_op = model(features, labels)
    eval_metrics = {
        'character': char_accuracy(predictions['predicted_ids'],
                                   labels['label']),
        'sequence': sequence_accuracy(predictions['predicted_ids'],
                                      labels['label'])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics)
  # Model parameters
  mparams = _create_model_params(dataset)
  # Estimator
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=estimator_config,
      params=mparams)

  train_hooks, eval_hooks = _create_hooks(mparams, output_dir)

  if FLAGS.schedule != 'train':
    # log to file
    file_name = "{}-tensorflow.log".format(mparams['dataset']['dataset_name'])
    file_name = os.path.join(FLAGS.output_dir, file_name)
    log = logging.getLogger('tensorflow')
    handle = logging.FileHandler(file_name)
    log.addHandler(handle)

  return Experiment(
      estimator=estimator,
      train_input_fn=dataset.create_input_fn,
      eval_input_fn=dataset.create_input_fn,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      train_monitors=train_hooks,
      eval_hooks=eval_hooks,
      eval_delay_secs=0)

def main(_argv):
  """Main function
  """
  schedules = ['train', 'evaluate', 'continuous_eval']
  assert FLAGS.schedule in schedules,\
                      "Only schedules: %s supported!"%(','.join(schedules))

  learn_runner.run(
      experiment_fn=_create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
