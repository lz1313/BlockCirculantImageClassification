import tensorflow as tf
slim = tf.contrib.slim
IMAGENET_MEAN = [103.939, 116.779, 123.68]


class Vgg16(object):
  """Implementation of the VGG16."""

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

    self.conv1_1 = conv(self.X, 3, 3, 64, 1, 1, name="conv1_1")
    self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, name="conv1_2")
    self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name="pool1")

    self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, name="conv2_1")
    self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, name="conv2_2")
    self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name="pool2")

    self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, name="conv3_1")
    self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, name="conv3_2")
    self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, name="conv3_3")
    self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, name="pool3")

    self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, name="conv4_1")
    self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, name="conv4_2")
    self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, name="conv4_3")
    self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, name="pool4")

    self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, name="conv5_1")
    self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, name="conv5_2")
    self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, name="conv5_3")
    self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, name="pool5")

    flattened = tf.reshape(self.pool5, [-1, 25088])
    self.fc6 = fc(
        flattened, 25088, 4096,
        name="fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    self.dropout6 = slim.dropout(
        self.fc6,
        self.KEEP_PROB,
        is_training=self.is_training,
        scope="dropout6")

    self.fc7 = fc(self.dropout6, 4096, 4096, name="fc7")

    self.dropout7 = slim.dropout(
        self.fc7,
        self.KEEP_PROB,
        is_training=self.is_training,
        scope="dropout7")

    self.logits = fc(
        self.dropout7, 4096, self.NUM_CLASSES, relu=False, name="fc8")


class Vgg19(object):
  """Implementation of the VGG19."""

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

    self.conv1_1 = conv(self.X, 3, 3, 64, 1, 1, name="conv1_1")
    self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, name="conv1_2")
    self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name="pool1")

    self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, name="conv2_1")
    self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, name="conv2_2")
    self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name="pool2")

    self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, name="conv3_1")
    self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, name="conv3_2")
    self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, name="conv3_3")
    self.conv3_4 = conv(self.conv3_3, 3, 3, 256, 1, 1, name="conv3_4")
    self.pool3 = max_pool(self.conv3_4, 2, 2, 2, 2, name="pool3")

    self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, name="conv4_1")
    self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, name="conv4_2")
    self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, name="conv4_3")
    self.conv4_4 = conv(self.conv4_3, 3, 3, 512, 1, 1, name="conv4_4")
    self.pool4 = max_pool(self.conv4_4, 2, 2, 2, 2, name="pool4")

    self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, name="conv5_1")
    self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, name="conv5_2")
    self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, name="conv5_3")
    self.conv5_4 = conv(self.conv5_3, 3, 3, 512, 1, 1, name="conv5_4")
    self.pool5 = max_pool(self.conv5_4, 2, 2, 2, 2, name="pool5")

    flattened = tf.reshape(self.pool5, [-1, 25088])
    self.fc6 = fc(
        flattened, 25088, 4096,
        name="fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    self.dropout6 = slim.dropout(
        self.fc6,
        self.KEEP_PROB,
        is_training=self.is_training,
        scope="dropout6")

    self.fc7 = fc(self.dropout6, 4096, 4096, name="fc7")

    self.dropout7 = slim.dropout(
        self.fc7,
        self.KEEP_PROB,
        is_training=self.is_training,
        scope="dropout7")

    self.logits = fc(
        self.dropout7, 4096, self.NUM_CLASSES, relu=False, name="fc8")


# def assign_from_npy_fn(model_path, skip_layers=[]):
#   """Returns a function that assigns specific variables from a checkpoint."""
#   with gfile.GFile(model_path, 'rb') as f:
#     weights_dict = np.load(f).item()
#     def callback(session):
#       # Loop over all layer names stored in the weights dict
#       for op_name in weights_dict:
#         if op_name not in skip_layers:
#           with tf.variable_scope(op_name, reuse=True):
#
#             # Assign weights/biases to their corresponding tf variable
#             for data in weights_dict[op_name]:
#
#               # Biases
#               if len(data.shape) == 1:
#                 var = tf.get_variable('biases', trainable=False)
#                 session.run(var.assign(data))
#
#               # Weights
#               else:
#                 var = tf.get_variable('weights', trainable=False)
#                 session.run(var.assign(data))
#               tf.logging.info("Restoring {}".format(var.op.name))
#     return callback
def conv(x,
         filter_height,
         filter_width,
         num_filters,
         stride_y,
         stride_x,
         name,
         padding="SAME",
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
        "weights",
        shape=[
            filter_height, filter_width, input_channels / groups, num_filters
        ])
    biases = tf.get_variable("biases", shape=[num_filters])

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
        "weights", shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable("biases", [num_out], trainable=True)

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
             padding="SAME"):
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
