import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import glob
from blockcirculant import  block_circulant
from net.alexnet import AlexNet
from net.vgg import Vgg16
from net.vgg import Vgg19
from net.lenet import LeNet
from net.cifarnet import CifarNet
import data_provider
import cifar_data_provider
import mnist_data_provider
from datetime import datetime
import time
slim = tf.contrib.slim
tf.app.flags.DEFINE_bool('use_admm', True,
                         'Whether to use admm to train the network.')
tf.app.flags.DEFINE_bool('retrain', False,
                         'Whether to retrain the network.')
tf.app.flags.DEFINE_bool('restore', True,
                         'Whether to restore from the checkpoint.')
tf.app.flags.DEFINE_string('checkpoint_path', '/data2/zli/admm_alexnet/baseline/model_epoch1000.ckpt',
                           'Path of checkpoint.')
tf.app.flags.DEFINE_string('train_dir', '/data2/zli/admm_alexnet/block_4',
                           'Path of trained model.')
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('admm_learning_rate', 0.0001,
                          """Initial admm learning rate.""")
tf.app.flags.DEFINE_float('dropout_rate', 0.66,
                          """Drop rate during training.""")
tf.app.flags.DEFINE_integer('block_size', 4, 'The block size')
tf.app.flags.DEFINE_string('network', 'CifarNet',
                           'Currently support LeNet, CifarNet, AlexNet, Vgg16 and Vgg19')
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           'Currently support flowers, mnist, cifar10 and imagenet')
tf.app.flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'The number of epochs to train.')

FLAGS = tf.flags.FLAGS

"""
Configuration Part.
"""
# How often we want to write the tf.summary data to disk
display_step = 100
summary_step = 100
admm_update_step = 10000
save_step = 10000

"""
Main Part of the finetuning Script.
"""
def main(_):
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
    # Create parent path if it doesn't exist
    if not os.path.isdir(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            # Place data loading and preprocessing on the cpu
            tr_data = ImageDataProvider(training_filenames,
                                        mode='training',
                                        batch_size=FLAGS.batch_size,
                                        num_classes=NUM_CLASSES,
                                        shuffle=True)
            val_data = ImageDataProvider(validation_filenames,
                                         mode='inference',
                                         batch_size=FLAGS.batch_size,
                                         num_classes=NUM_CLASSES,
                                         shuffle=False)

        x = tf.placeholder(tf.float32, INPUT_SHAPE)
        y = tf.placeholder(tf.float32, [FLAGS.batch_size, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)

        model = None
        # Initialize model
        if FLAGS.network == 'AlexNet':
            model = AlexNet(x, keep_prob, NUM_CLASSES, True)
        elif FLAGS.network == 'Vgg16':
            model = Vgg16(x, keep_prob, NUM_CLASSES, True)
        elif FLAGS.network == 'Vgg19':
            model = Vgg19(x, keep_prob, NUM_CLASSES, True)
        elif FLAGS.network == 'LeNet':
            model = LeNet(x, keep_prob, NUM_CLASSES, True)
        elif FLAGS.network == 'CifarNet':
            model = CifarNet(x, keep_prob, NUM_CLASSES, True)
        else:
            print("Model not supported")
            exit()
        # Link variable to model output
        score = model.logits
        # List of trainable variables of the layers we want to train
        # var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        var_list = [v for v in tf.trainable_variables()]

        if FLAGS.use_admm:
        # admm update
        # Iterate all the weights.
            count = 0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if 'conv' in v.name and 'conv1' not in v.name:
                        u = tf.get_variable(
                            'U_{}'.format(count),
                            shape=v.get_shape().as_list(),
                            trainable=False,
                            initializer=tf.initializers.zeros)
                        z = tf.get_variable(
                            'Z_{}'.format(count),
                            shape=v.get_shape().as_list(),
                            trainable=False)
                        weight_shape, indices = block_circulant.generate_shape_and_indices(
                            v.get_shape().as_list()[2],
                            v.get_shape().as_list()[3],
                            kernel_h=v.get_shape().as_list()[0],
                            kernel_w=v.get_shape().as_list()[1],
                            block_size=FLAGS.block_size)
                        make_bc = tf.py_func(block_circulant.make_block_circulant,
                                             [v + u, weight_shape, indices], tf.float32)
                        make_bc_ref = tf.py_func(block_circulant.make_block_circulant,
                                                 [v, weight_shape, indices], tf.float32)
                        assign_z_op = z.assign(make_bc)
                        assign_u_op = u.assign(u + v - z)
                        assign_v_op = v.assign(make_bc_ref)
                        tf.add_to_collection('ASSIGN_U_OP', assign_u_op)
                        tf.add_to_collection('ASSIGN_Z_OP', assign_z_op)
                        tf.add_to_collection('ASSIGN_V_OP', assign_v_op)
                        tf.add_to_collection('MSE', tf.nn.l2_loss(u + v - z))
                        count += 1

            with tf.name_scope('admm_update'):
                update_ops_u = tf.get_collection('ASSIGN_U_OP')
                update_ops_z = tf.get_collection('ASSIGN_Z_OP')
                update_ops_v = tf.get_collection('ASSIGN_V_OP')
                l2_losses = tf.get_collection('MSE')

        # Specify the loss function:
        with tf.name_scope("cross_ent"):
            cel = tf.losses.softmax_cross_entropy(
                onehot_labels=y, logits=score, label_smoothing=0.0, weights=1.0)
            if FLAGS.use_admm:
                l2_loss = tf.reduce_mean(l2_losses)
                total_loss = cel + FLAGS.admm_learning_rate * l2_loss
                slim.summaries.add_scalar_summary(total_loss, 'Total_Loss', 'losses')
                slim.summaries.add_scalar_summary(l2_loss, 'MSE_Loss', 'losses')
                slim.summaries.add_scalar_summary(cel, 'CEL_Loss', 'losses')
            else:
                total_loss = cel
                slim.summaries.add_scalar_summary(total_loss, 'Total_Loss', 'losses')

        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            # gradients = tf.gradients(loss, var_list)
            # gradients = list(zip(gradients, var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_tensor = slim.learning.create_train_op(
                total_loss,
                optimizer=opt,
                update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

            if FLAGS.retrain and FLAGS.use_admm:
                grads_and_vars = opt.compute_gradients(cel)
                new_grads_and_vars = []
                for grad_and_var in grads_and_vars:
                    grad, v = grad_and_var
                    if len(v.get_shape().as_list()) == 4:
                        weight_shape, indices = block_circulant.generate_shape_and_indices(
                            v.get_shape().as_list()[2],
                            v.get_shape().as_list()[3],
                            kernel_h=v.get_shape().as_list()[0],
                            kernel_w=v.get_shape().as_list()[1],
                            block_size=FLAGS.block_size)
                        new_grad = tf.py_func(block_circulant.make_block_circulant,
                                              [grad, weight_shape, indices], tf.float32)
                    else:
                        new_grad = grad
                    new_grads_and_vars.append((new_grad, v))

                grad_updates = opt.apply_gradients(new_grads_and_vars)
                retrain_tensor = control_flow_ops.with_dependencies([grad_updates], cel)

        # Summaries:
        slim.summaries.add_histogram_summaries(slim.get_model_variables())
        slim.summaries.add_scalar_summary(FLAGS.learning_rate, 'Learning_Rate',
                                          'training')
        train_op = {}
        if FLAGS.use_admm:
            if FLAGS.retrain:
                train_op['update_v'] = update_ops_v
                train_op['loss'] = retrain_tensor
            else:
                train_op['loss'] = train_tensor
                train_op['Z'] = update_ops_z
                train_op['U'] = update_ops_u
        else:
            train_op['loss'] = train_tensor
        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)

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
        writer = tf.summary.FileWriter(FLAGS.train_dir)

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(TRAIN_SIZE / FLAGS.batch_size))
        val_batches_per_epoch = int(np.floor(VAL_SIZE / FLAGS.batch_size))

        # Start Tensorflow session
        with tf.Session() as sess:

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)

            if FLAGS.restore:
                # Load the pretrained weights
                saver.restore(sess, FLAGS.checkpoint_path)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              FLAGS.train_dir))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("{} Dataset initialized.".format(datetime.now()))
            if FLAGS.use_admm:
                if FLAGS.retrain:
                    _ = sess.run( train_op['update_v'])
                else:
                    _ = sess.run(train_op['Z'])
            # Loop over number of epochs
            for epoch in range(FLAGS.num_epochs):

                print("{} Epoch number: {}".format(datetime.now(), epoch+1))

                for step in range(train_batches_per_epoch * epoch, train_batches_per_epoch * (epoch + 1)):
                    # get next batch of data
                    start_time = time.clock()
                    img_batch, label_batch = sess.run([tr_data.images, tr_data.labels])
                    elapsed_time = time.clock() - start_time

                    # And run the training op
                    start_time = time.clock()
                    loss_value = sess.run(train_op['loss'],feed_dict={x: img_batch,
                                                                      y: label_batch,
                                                                      keep_prob: FLAGS.dropout_rate})

                    # display loss
                    if step % display_step == 0:
                        elapsed_time = time.clock() - start_time
                        print('step:{} ({} s/step), loss = {:4f}'.format(step, elapsed_time/float(display_step),
                                                                         loss_value))
                    if FLAGS.use_admm:
                        # admm_update
                        if not FLAGS.retrain:
                            if (step + 1) % admm_update_step == 0:
                                start_time = time.clock()
                                _ = sess.run(
                                    train_op['Z'])
                                _ = sess.run(
                                    train_op['U'])
                                elapsed_time = time.clock() - start_time
                                print('admm step:{} s/step'.format(elapsed_time))

                    # Generate summary with the current batch of data and write to file
                    if (step + 1) % summary_step == 0:
                        s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                                y: label_batch,
                                                                keep_prob: 1.})

                        writer.add_summary(s, epoch*train_batches_per_epoch + step)

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))

                test_acc = 0.
                test_acc_5 = 0.
                test_count = 0
                start = time.clock()
                for bc in range(val_batches_per_epoch):
                    img_batch, label_batch = sess.run([val_data.images, val_data.labels])

                    acc, acc_5 = sess.run([accuracy, top_5_accuracy], feed_dict={x: img_batch,
                                                                                 y: label_batch,
                                                                                 keep_prob: 1.})
                    print("Batch {}/{}, acc {:4f}, recall@5 {:4f}".format(bc + 1, val_batches_per_epoch, acc, acc_5))
                    test_acc += acc
                    test_acc_5 += acc_5
                    test_count += 1
                test_acc /= test_count
                test_acc_5 /= test_count
                print("{} Validation Accuracy = {:.4f} Validation Recall@5 = {:.4f}".format(datetime.now(), test_acc,
                                                                                            test_acc_5))

                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(FLAGS.train_dir,
                                               'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                               checkpoint_name))

            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    tf.app.run()