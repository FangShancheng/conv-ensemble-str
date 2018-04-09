# Copyright 2017 IIE, CAS.
# Written by Shancheng Fang
# ==============================================================================

"""Define flags are common for train_eval.py scripts."""
import tensorflow as tf


def define():
  """Define common flags."""
  # yapf: disable
  tf.flags.DEFINE_string("output_dir", "/tmp/workdir",
                         """The directory to write model checkpoints and
                         summaries. If None, a local temporary directory
                         is created.""")
  tf.flags.DEFINE_string("checkpoint", None,
                         """checkpoint to restore variables""")
  tf.flags.DEFINE_boolean("debug", False,
                          """use tfdbg to debug""")

  # Model config
  tf.flags.DEFINE_integer("beam_width", 5,
                          """beam width. 0 for close beam search.""")

  # Model hyper parameters
  tf.flags.DEFINE_string("optimizer", "Momentum",
                         """the optimizer to use""")
  tf.flags.DEFINE_float("learning_rate", 0.01,
                        """learning rate""")
  tf.flags.DEFINE_float("clip_gradients", 20.0,
                        """number of clipped gradients""")
  tf.flags.DEFINE_float("momentum", 0.9,
                        """momentum value for the momentum optimizer if
                        used""")
  tf.flags.DEFINE_boolean("use_nesterov", True,
                          """use nesterov""")

  # Dataset config
  tf.flags.DEFINE_string("dataset_name", "MJSynth",
                         """Name of the dataset. Supported: fsns""")
  tf.flags.DEFINE_string("dataset_dir", None,
                         """Dataset root folder.""")
  tf.flags.DEFINE_string("split_name", None,
                         """Name of the dataset split.""")
  tf.flags.DEFINE_integer("batch_size", 128,
                          """Batch size used for training and evaluation.""")

  # Training and evaluating parameters
  tf.flags.DEFINE_string("schedule", "train",
                         """Estimator function to call, defaults to
                         continuous_train_and_eval for local run""")
  tf.flags.DEFINE_integer("train_steps", 1000000,
                          """Maximum number of training steps to run.
                           If None, train forever.""")
  tf.flags.DEFINE_integer("eval_steps", 500,
                          "Run N steps evaluation.")

  # RunConfig Flags
  tf.flags.DEFINE_integer("tf_random_seed", None,
                          """Random seed for TensorFlow initializers. Setting
                          this value allows consistency between reruns.""")
  tf.flags.DEFINE_integer("save_checkpoints_secs", 900,
                          """Save checkpoints every this many seconds.
                          Can not be specified with save_checkpoints_steps.""")
  tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                          """Save checkpoints every this many steps.
                          Can not be specified with save_checkpoints_secs.""")
  tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                          """Maximum number of recent checkpoint files to keep.
                          As new files are created, older files are deleted.
                          If None or 0, all checkpoint files are kept.""")
  tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                          """In addition to keeping the most recent checkpoint
                          files, keep one checkpoint file for every N hours of
                          training.""")
  tf.flags.DEFINE_float("gpu_memory_fraction", 1.0,
                        """Fraction of GPU memory used by the process on
                        each GPU uniformly on the same machine.""")
  tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                          """Allow GPU memory allocation to grow
                          dynamically.""")
  tf.flags.DEFINE_integer("log_step", 100,
                          """log_step_count_steps""")

  # Summary config
  tf.flags.DEFINE_boolean("summary", True,
                          """log to summary""")
  tf.flags.DEFINE_integer("max_outputs", 4,
                          """the max outputs number to summary images and text
                          in a batch""")
  # yapf: enable
