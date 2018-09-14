import tensorflow as tf

slim = tf.contrib.slim

class CifarNet(object):
  """Implementation of the AlexNet."""

  def __init__(self, x, keep_prob, num_classes, is_training):
    """Create the graph of the AlexNet model.
    Args:
      x: Placeholder for the input tensor.
      keep_prob: Dropout probability.
      num_classes: Number of classes in the dataset.
      is_training: is_training.
    """
    # Parse input arguments into class variables
    self.X = x
    self.logits = None
    self.KEEP_PROB = keep_prob
    self.NUM_CLASSES = num_classes
    self.is_training = is_training
    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):
    """Create the network graph."""
    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    conv1 = conv(self.X, 5, 5, 64, 1, 1, padding='VALID', name='conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(pool1, 4, 1e-3/9.0, 0.75, name='norm1')


    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool
    conv2 = conv(norm1, 5, 5, 64, 1, 1, name='conv2')
    norm2 = lrn(conv2, 4, 1e-3/9.0, 0.75, name='norm2')
    pool2 = max_pool(norm2, 2, 2, 2, 2, padding='VALID', name='pool2')

    # 3th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc3 = fc(flattened, 7 * 7 * 64, 384, name='fc3')
    dropout3 = slim.dropout(
        fc3, self.KEEP_PROB, is_training=self.is_training, scope='dropout3')

    # 4th Layer: FC (w ReLu) -> Dropout
    fc4 = fc(dropout3, 384, 192, name='fc4')
    dropout4 = slim.dropout(
        fc4, self.KEEP_PROB, is_training=self.is_training, scope='dropout4')

    # 5th Layer: FC and return unscaled activations
    self.logits = fc(dropout4, 192, self.NUM_CLASSES, relu=False, name='fc8')


def conv(x,
         filter_height,
         filter_width,
         num_filters,
         stride_y,
         stride_x,
         name,
         padding='SAME',
         groups=1):
  """Create a convolution layer.
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                     strides=[1, stride_y, stride_x, 1],
                     padding=padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable(
        'weights',
        shape=[
            filter_height, filter_width, input_channels / groups, num_filters
        ])
    biases = tf.get_variable('biases', shape=[num_filters])

  if groups == 1:
    conv = convolve(x, weights)

  # In the cases of multiple groups, split inputs & weights and
  else:
    # Split input and weights and convolve them separately
    input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
    weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
    output_groups = [
        convolve(i, k) for i, k in zip(input_groups, weight_groups)
    ]

    # Concat the convolved output together again
    conv = tf.concat(axis=3, values=output_groups)

  # Add biases
  bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

  # Apply relu function
  relu = tf.nn.relu(bias, name=scope.name)

  return relu


def fc(x, num_in, num_out, name, relu=True):
  """Create a fully connected layer."""
  with tf.variable_scope(name) as scope:

    # Create tf variables for the weights and biases
    weights = tf.get_variable(
        'weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

  if relu:
    # Apply ReLu non linearity
    relu = tf.nn.relu(act)
    return relu
  else:
    return act


def max_pool(x,
             filter_height,
             filter_width,
             stride_y,
             stride_x,
             name,
             padding='SAME'):
  """Create a max pooling layer."""
  return tf.nn.max_pool(
      x,
      ksize=[1, filter_height, filter_width, 1],
      strides=[1, stride_y, stride_x, 1],
      padding=padding,
      name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
  """Create a local response normalization layer."""
  return tf.nn.local_response_normalization(
      x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
  """Create a dropout layer."""
  return tf.nn.dropout(x, keep_prob)
