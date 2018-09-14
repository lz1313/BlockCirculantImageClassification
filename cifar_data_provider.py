import tensorflow as tf
num_readers = 1
num_preprocess_threads = 1
_PADDING = 4

class ImageDataProvider(object):
    def __init__(self, filenames, num_classes, batch_size, mode, shuffle=True):
        self.num_classes = num_classes
        self.filenames = filenames
        self.is_training = mode == 'training'
        if mode == 'training':
            filename_queue = tf.train.string_input_producer(self.filenames,
                                                            shuffle=shuffle,
                                                            capacity=16,
                                                            name='filename_queue')
        else:
            filename_queue = tf.train.string_input_producer(self.filenames,
                                                            shuffle=shuffle,
                                                            capacity=1,
                                                            name='filename_queue')
        # Approximate number of examples per shard.
        examples_per_shard = 1024
        if mode == 'training':
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = (examples_per_shard * 4)
            capacity = min_queue_examples + 4 * batch_size
            examples_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string], name='random_examples_queue')
        else:
            capacity = examples_per_shard + 4 * batch_size
            examples_queue = tf.FIFOQueue(
                capacity=capacity,
                dtypes=[tf.string], name='fifo_examples_queue')

        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_embedding = self._parse_function(example_serialized)
            images_and_labels.append([image_buffer, label_embedding])

        images, label_embedding_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=4 * num_preprocess_threads * batch_size,
            name='batch_join')
        self.images = images
        self.labels = label_embedding_batch

    def preprocess_for_train(self, image,
                             output_height,
                             output_width,
                             padding=_PADDING,
                             add_image_summaries=True):
        """Preprocesses the given image for training.
        Note that the actual resizing scale is sampled from
          [`resize_size_min`, `resize_size_max`].
        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          padding: The amound of padding before and after each dimension of the image.
          add_image_summaries: Enable image summaries.
        Returns:
          A preprocessed image.
        """
        if add_image_summaries:
            tf.summary.image('image', tf.expand_dims(image, 0))

        # Transform the image to floats.
        image = tf.to_float(image)
        if padding > 0:
            image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(image,
                                         [output_height, output_width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        if add_image_summaries:
            tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        return tf.image.per_image_standardization(distorted_image)

    def preprocess_for_eval(self, image, output_height, output_width,
                            add_image_summaries=True):
        """Preprocesses the given image for evaluation.
        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          add_image_summaries: Enable image summaries.
        Returns:
          A preprocessed image.
        """
        if add_image_summaries:
            tf.summary.image('image', tf.expand_dims(image, 0))
        # Transform the image to floats.
        image = tf.to_float(image)

        # Resize and crop if needed.
        resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                               output_width,
                                                               output_height)
        if add_image_summaries:
            tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

        # Subtract off the mean and divide by the variance of the pixels.
        return tf.image.per_image_standardization(resized_image)

    def preprocess_image(self, image, output_height, output_width, is_training=False,
                         add_image_summaries=True):
        """Preprocesses the given image.
        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.
          add_image_summaries: Enable image summaries.
        Returns:
          A preprocessed image.
        """
        if is_training:
            return self.preprocess_for_train(
                image, output_height, output_width,
                add_image_summaries=add_image_summaries)
        else:
            return self.preprocess_for_eval(
                image, output_height, output_width,
                add_image_summaries=add_image_summaries)

    def _parse_function(self, example_proto):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], dtype=tf.int64, default_value=-1),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        img_string = parsed_features['image/encoded']
        label = parsed_features['image/class/label']
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # load and preprocess the image
        image = tf.image.decode_png(img_string, channels=3)
        image = self.preprocess_image(image, 32, 32, self.is_training)
        return image, one_hot