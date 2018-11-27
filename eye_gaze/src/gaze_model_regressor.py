""" Script for training the source model """

from __future__ import print_function

from model.ops import *
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from dataio.data_reader import CLSReader
from model.base_model import SimpleModel, MeanReLUModel

F = tf.app.flags.FLAGS
r2d = 180.0/3.14

# Mean of synthetic and real data, computed offline
syn_data_mean = 110.
real_data_mean = 125.

class GazeRegressor():
    def __init__(self):
        # Initiliaze the data reader
        self.dataloader = CLSReader()

        # Construct the model architecture to be learned
        self.build_model()

        self.training_data = self.dataloader.create_training_dataset()
        self.validation_data = self.dataloader.create_validation_dataset()

        self.training_iter = self.training_data.make_one_shot_iterator()
        self.validation_iter = self.validation_data.make_initializable_iterator()

    def get_loss(self):
        """ Attach the relevant loss depending on the type of loss selected """

        if F.loss_type=="cosine":
            self.angle_loss = r2d*tf.acos(1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.labels,1), tf.nn.l2_normalize(self.out, 1), dim=1))
            self.loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.labels,1), tf.nn.l2_normalize(self.out, 1), dim=1)

        elif F.loss_type=="mse2d":
            xl, yl, zl = tf.split(self.labels, 3, axis=1)
            xo, yo, zo = tf.split(self.out, 3, axis=1)
            thetal, thetao = tf.asin(-yl), tf.asin(-yo)
            phil, phio = tf.atan2(-zl, -xl), tf.atan2(-zo, -xo)
            self.lb = tf.concat([thetal, phil], axis=1)
            self.ob = tf.concat([thetao, phio], axis=1)
            self.loss = tf.scalar_mul(tf.constant(r2d), tf.losses.mean_squared_error(self.lb, self.ob, 2))
        
        elif F.loss_type=="mse3d":
            self.loss = tf.losses.mean_squared_error(tf.nn.l2_normalize(self.labels, 0), tf.nn.l2_normalize(self.out, 0))

    def build_model(self):
        """ Loads the model from model/base_model.py """

        self.images, self.labels = self.dataloader.get_model_inputs()

        self.mean = tf.placeholder_with_default(syn_data_mean, shape=())
        self.prob = tf.placeholder_with_default(0.5, shape=())

        model = MeanReLUModel(self.images, self.labels, output_dim=F.output_dim, mean=self.mean, prob=self.prob, scope='source_regressor')
        self.out, _ = model.get_model()
        self.get_loss()        

    def train(self):
        """ The actual training process """

        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Training Source Network")

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Learning rate with an exponential decay
        self.lr = tf.train.exponential_decay(
            learning_rate=F.initial_learning_rate,
            global_step=global_step,
            decay_steps=F.decay_step,
            decay_rate=F.decay_rate,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        self.grad_update = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=global_step)

        # Attaching tf variable to summary for visualisation using tensorboard
        tf.summary.scalar('mse_loss', self.loss)
        tf.summary.scalar('learning_rate', self.lr)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=None)

        # Reload a saved checkpoint
        def restore_fn(sess):
            return self.saver.restore(sess, F.checkpoint_dir + F.checkpoint_file)

        self.training_handle_op = self.training_iter.string_handle()
        self.validation_handle_op = self.validation_iter.string_handle()

        # Define your supervisor for running a managed session.
        if F.load_chkpt:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, init_fn=restore_fn, saver=self.saver)
        else:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, init_fn=None, saver=self.saver)

        current_best_loss = 1000.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.training_handle = sess.run(self.training_handle_op)
            self.validation_handle = sess.run(self.validation_handle_op)

            for step in range(int(F.num_steps)):
                try:
                    if step % F.log_every == 0:
                        loss, angle_loss,  _, summaries, global_step_count = sess.run([self.loss, self.angle_loss, self.grad_update,
                         self.summary_op, sv.global_step], 
                         feed_dict={self.dataloader.split_handle: self.training_handle, self.mean: syn_data_mean,
                         self.prob: 0.5})
                        sv.summary_computed(sess, summaries, global_step=global_step_count)
                        logging.info("Step: {}/{}, Global Step: {}, loss: {}, Angle Loss: {}".format(step, F.num_steps, global_step_count, loss, angle_loss))
                    else:
                        loss, _, global_step_count = sess.run([self.loss, self.grad_update,
                         sv.global_step], feed_dict={self.dataloader.split_handle: self.training_handle})
                except:
                    logging.info("Smaller batch size error,.. proceeding to next batch size")

                # print current performance on validation set to monitor training
                if step % F.save_every==1:
                    logging.info('Saving model to disk as step={}'.format(step))
                    sess.run(self.validation_iter.initializer)
                    eval_loss = []
                    while True:
                        try:
                            loss, labels = sess.run([self.angle_loss, self.labels], 
                                feed_dict={self.dataloader.split_handle: self.validation_handle, self.mean: syn_data_mean,
                                self.prob: 0.0})
                            eval_loss.append(loss)
                        except:
                            if len(eval_loss) != 0:
                                eval_loss = np.array(eval_loss)
                                logging.info("Current Evaluation Loss at step({}): {}, Mean Angle Loss: {}, {}, {}".format(step, len(eval_loss), eval_loss.mean(), eval_loss.max(), eval_loss.min()))
                            break
