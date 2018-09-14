import tensorflow as tf
num_readers = 1
num_preprocess_threads = 1
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

    def _parse_function(self, example_proto):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='jpeg'),
            'image/class/label': tf.FixedLenFeature(
                [], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
            'image/object/bbox/xmin': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(
                dtype=tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        img_string = parsed_features['image/encoded']
        label = parsed_features['image/class/label']
        # text = parsed_features['image/class/text']
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes + 1)
        one_hot = one_hot[1:]
        # load and preprocess the image
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        imagenet_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        img_centered = tf.subtract(img_resized, imagenet_mean)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot