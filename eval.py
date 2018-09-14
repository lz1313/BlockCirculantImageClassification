import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import tensorflow as tf
import glob
from net.alexnet import AlexNet
from net.vgg import Vgg16
from net.vgg import Vgg19
from net.lenet import LeNet
from net.cifarnet import CifarNet
import data_provider
import cifar_data_provider
import mnist_data_provider
from datetime import datetime

tf.app.flags.DEFINE_string('checkpoint_path', '/data2/zli/pretrained_tf_models/alexnet/bvlc_alexnet.ckpt',
                         'Path of checkpoint.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'The number of samples in each batch.')
FLAGS = tf.flags.FLAGS
TRAIN_SIZE = 1281167
VAL_SIZE = 50000
"""
Configuration Part.
"""
# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"

if FLAGS.dataset == 'flowers':
    training_filenames = glob.glob('/data2/zli/flowers/*train-*-*')
    validation_filenames = glob.glob('/data2/zli/flowers/*validation-*-*')
    ImageDataProvider = cifar_data_provider.ImageDataProvider
    NUM_CLASSES = 5
    INPUT_SHAPE = [FLAGS.batch_size, 32, 32, 3]
    TRAIN_SIZE = 3320
    VAL_SIZE = 350
elif FLAGS.dataset == 'mnist':
    training_filenames = glob.glob('/data2/zli/mnist/*train*')
    validation_filenames = glob.glob('/data2/zli/mnist/*test*')
    ImageDataProvider = mnist_data_provider.ImageDataProvider
    NUM_CLASSES = 10
    INPUT_SHAPE = [FLAGS.batch_size, 28, 28, 1]
    TRAIN_SIZE = 60000
    VAL_SIZE = 10000
elif FLAGS.dataset == 'cifar10':
    training_filenames = glob.glob('/data2/zli/cifar10/*train*')
    validation_filenames = glob.glob('/data2/zli/cifar10/*test*')
    ImageDataProvider = cifar_data_provider.ImageDataProvider
    NUM_CLASSES = 10
    INPUT_SHAPE = [FLAGS.batch_size, 32, 32, 3]
    TRAIN_SIZE = 50000
    VAL_SIZE = 10000
elif FLAGS.dataset == 'imagenet':
    training_filenames = glob.glob('/data2/siyu/train-*-*')
    validation_filenames = glob.glob('/data2/siyu/validation-*-*')
    ImageDataProvider = data_provider.ImageDataProvider
    NUM_CLASSES = 1000
    INPUT_SHAPE = [FLAGS.batch_size, 227, 227, 3]
    TRAIN_SIZE = 1281167
    VAL_SIZE = 50000
else:
    print("Dataset not supported")
    exit()

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    val_data = ImageDataProvider(validation_filenames,
                                  mode='inference',
                                  batch_size=FLAGS.batch_size,
                                  num_classes=NUM_CLASSES,
                                  shuffle=False)
    # val_iterator = val_data.iterator
    # val_next_batch = val_iterator.get_next()

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [FLAGS.batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [FLAGS.batch_size, NUM_CLASSES])
# x = val_data.images
# y = val_data.labels
keep_prob = tf.placeholder(tf.float32)

# Initialize model
if FLAGS.network == 'AlexNet':
    model = AlexNet(x, keep_prob, NUM_CLASSES, False)
elif FLAGS.network == 'Vgg16':
    model = Vgg16(x, keep_prob, NUM_CLASSES, False)
elif FLAGS.network == 'Vgg19':
    model = Vgg19(x, keep_prob, NUM_CLASSES, False)
elif FLAGS.network == 'LeNet':
    model = LeNet(x, keep_prob, NUM_CLASSES, False)
elif FLAGS.network == 'CifarNet':
    model = CifarNet(x, keep_prob, NUM_CLASSES, False)
else:
    print("Model not supported")
    exit()

# Link variable to model output
score = tf.nn.softmax(model.logits)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    correct_pred_5 = tf.nn.in_top_k(predictions=score, targets=tf.argmax(y, 1), k=5)
    top_5_accuracy = tf.reduce_mean(tf.cast(correct_pred_5, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('Recall@5', top_5_accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of validation steps per epoch
val_batches_per_epoch = int(np.floor(VAL_SIZE / FLAGS.batch_size))



# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights
    saver.restore(sess, FLAGS.checkpoint_path)


    # Validate the model on the entire validation set
    print("{} Start validation".format(datetime.now()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Initialize `iterator` with training data.
    # validation_filenames = glob.glob('/data2/siyu/validation-*-*')
    # sess.run(val_iterator.initializer, feed_dict={filenames: validation_filenames})
    test_acc = 0.
    test_acc_5 = 0.
    test_count = 0
    # fig = plt.figure(figsize=(30, 6))
    start = time.clock()
    for bc in range(val_batches_per_epoch):
        img_batch, label_batch = sess.run([val_data.images, val_data.labels])
        # class_name = class_names[np.argmax(np.squeeze(label_batch))]
        # print(np.argmax(np.squeeze(label_batch)))
        # fig.add_subplot(1, 6, bc + 1)
        # plt.imshow(cv2.cvtColor(np.squeeze(img_batch)/255.0 + 0.5, cv2.COLOR_BGR2RGB))
        # plt.title("Class: " +  np.squeeze(text_batch))
        # plt.axis('off')
        # result = Image.fromarray((np.squeeze(img_batch)).astype(np.uint8))
        # result.save('/tmp/{}.bmp'.format(bc))
        acc, acc_5= sess.run([accuracy, top_5_accuracy], feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: 1.})
        # acc, acc_5 = sess.run([accuracy, top_5_accuracy], feed_dict={keep_prob: 1.})
        print("Batch {}/{}, acc {:4f}, recall@5 {:4f}".format(bc + 1, val_batches_per_epoch, acc, acc_5))
        test_acc += acc
        test_acc_5 += acc_5
        test_count += 1
    # plt.show()
    print('Elapsed time: {:2f}'.format((time.clock() - start) / 60.))
    test_acc /= test_count
    test_acc_5 /= test_count
    print("{} Validation Accuracy = {:.4f} Validation Recall@5 = {:.4f}".format(datetime.now(), test_acc, test_acc_5))
    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()
