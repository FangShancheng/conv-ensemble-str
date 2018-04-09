# -*- coding: utf-8 -*-
import functools
import re
import tensorflow as tf
import inception_preprocessing

class CharsetMapper(object):
  """A simple class to map tensor ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.

    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

  def __init__(self,
               filename,
               max_sequence_length=30,
               default_character='?',
               null_character=u'\u2591'):
    """Creates a lookup table.

    Args:
      charset: a dictionary with id-to-character mapping.
    """
    self.null_character = null_character
    self.charset = self._read_charset(filename)
    self.max_sequence_length = max_sequence_length

    charset_array = self._dict_to_array(self.charset, default_character)
    mapping_strings = tf.constant(charset_array)
    self.table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=mapping_strings, default_value=default_character)
    self.invert_table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings)

  @property
  def num_charset(self):
    return len(self.charset)

  def get_text(self, ids):
    """ Returns a string corresponding to a sequence of character ids.

        Args:
          ids: a tensor with shape [batch_size, max_sequence_length]
    """
    return tf.reduce_join(
        self.table.lookup(tf.to_int64(ids)), reduction_indices=1)

  def get_label(self, text, null_character=u'\u2591'):
    """ Returns the ids of the corresponding text,

        Args:
          text: a tensor with shape [batch_size, lexicon_size]
                         and type string
          null_character: a unicode character used to replace '<null>'
          character. the default value is a light shade block '░'.
    """
    batch_size = text.shape[0].value
    lexicon_size = text.shape[1].value
    text = tf.reshape(text, [-1])
    sp_text = tf.string_split(text, delimiter='')
    sp_text = tf.sparse_reset_shape(sp_text, [batch_size*lexicon_size,
                                              self.max_sequence_length])
    sp_text = tf.sparse_tensor_to_dense(sp_text, default_value=null_character)
    ids = self.invert_table.lookup(sp_text)
    ids = tf.reshape(ids, [batch_size, lexicon_size, self.max_sequence_length])
    return tf.to_int32(ids)

  def _dict_to_array(self, id_to_char, default_character):
    num_char_classes = max(id_to_char.keys()) + 1
    array = [default_character] * num_char_classes
    for k, v in id_to_char.iteritems():
      array[k] = v
    return array

  def _read_charset(self, filename):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the FSNS dataset.

    Args:
      filename: a path to the charset file.

    Returns:
      a dictionary with keys equal to character codes and values - unicode
      characters.
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
      for i, line in enumerate(f):
        m = pattern.match(line)
        if m is None:
          tf.logging.warning('incorrect charset file. line #%d: %s', i, line)
          continue
        code = int(m.group(1))
        char = m.group(2).decode('utf-8')
        if char == '<nul>':
          char = self.null_character
        charset[code] = char
    return charset

def augment_image(image):
  """Augmentation the image with a random modification.

  Args:
    image: input Tensor image of rank 3, with the last dimension
           of size 3.

  Returns:
    Distorted Tensor image of the same shape.
  """
  with tf.variable_scope('AugmentImage'):
    height = image.get_shape().dims[0].value
    width = image.get_shape().dims[1].value

    # Random crop cut from the street sign image, resized to the same size.
    # Assures that the crop is covers at least 0.8 area of the input image.
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4]),
        min_object_covered=0.8,
        aspect_ratio_range=[0.8, 1.2],
        area_range=[0.8, 1.0],
        use_image_if_no_bounding_boxes=True)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # Randomly chooses one of the 4 interpolation methods
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=4)
    distorted_image.set_shape([height, width, 3])

    # Color distortion
    # TODO:incompatible with clip value in inception_preprocessing.distort_color
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        functools.partial(
            inception_preprocessing.distort_color, fast_mode=False),
        num_cases=4)
    distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)

  return distorted_image
