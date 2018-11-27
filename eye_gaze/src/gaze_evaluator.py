""" Script to evaluate the model performance """

from __future__ import print_function

from model.ops import *
import random
import cv2
from PIL import Image
import os, sys
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from dataio.data_reader import CLSReader
from model.base_model import SimpleModel
F = tf.app.flags.FLAGS
r2d = 180.0/3.14

class GazeEval():
    def __init__(self):
        # Initiliaze the data reader
        self.dataloader = CLSReader()

        # Construct the model architecture to be learned
        self.build_model()

        self.validation_data = self.dataloader.create_validation_dataset()
        self.validation_iter = self.validation_data.make_initializable_iterator()

    def get_loss(self):
        """ Attach the relevant loss depending on the type of loss selected """

        if F.loss_type=="cosine":
            self.losscos = r2d*tf.acos(1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.labels,1), tf.nn.l2_normalize(self.out, 1), dim=1))
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

        model = SimpleModel(self.images, self.labels, output_dim=F.output_dim, scope='source_regressor')
        self.out, _ = model.get_model()
        self.get_loss()

    def eval(self):
        """ The evaluation process """

        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Testing Source Network")

        # Attaching tf variable for Tensorboard visualisation
        tf.summary.scalar('mse_loss', self.loss)
        tf.summary.scalar('angle_loss', self.losscos)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=None)

        # Load the trained model for evaluation
        def restore_fn(sess):
            print("Log-dir: ", F.checkpoint_dir)
            return self.saver.restore(sess, F.checkpoint_dir + F.checkpoint_file)

        self.validation_handle_op = self.validation_iter.string_handle()

        # Define your supervisor for running a managed session.
        sv = tf.train.Supervisor(logdir=F.log_eval_dir, init_fn=restore_fn, summary_op=None, saver=self.saver)

        current_best_loss = 1000. #TODO: Read it from a file for multiple restarts
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    
            logging.info('Starting evaluation: ')
            self.validation_handle = sess.run(self.validation_handle_op)
            sess.run(self.validation_iter.initializer)
            eval_loss = []
            while True:
                try:
                    if not F.visualise:
                        loss, losscos = sess.run([self.loss, self.losscos], feed_dict={self.dataloader.split_handle: self.validation_handle})
                        logging.info("Batch loss3d: {}, losscos: {}".format(loss, losscos))
                        eval_loss.append(losscos)
                    else:
                        loss, images, gts, preds, gts_labels, losscos = sess.run([self.loss, self.images, self.lbl, self.out, self.labels, self.losscos])
                        images = np.uint8(images)
                        for idx in range(len(images)):
                            eye_c = np.array([55/2, 35/2])
                            cv2.line(images[idx], tuple(eye_c), tuple(eye_c+(gts[idx,:2]*80).astype(int)), (0, 127, 127), 1)
                            cv2.line(images[idx], tuple(eye_c), tuple(eye_c+(preds[idx,:2]*80).astype(int)), (255,255,255), 1)                                
                            im = Image.fromarray(images[idx].reshape(35, 55))
                            a_loss = self.angle_disparity(gts[idx], preds[idx])
                            im.save(F.visualise_dir + os.sep + str(idx) + '__' +  str(a_loss) + '.jpg')
                            print("Gt:", gts[idx], "||", "  Pred: ", preds[idx], "||", "aloss: ", a_loss)
                        print("Vis done.")
                        sys.exit()
                except:
                    print("Exception Raised")
                    eval_loss = np.array(eval_loss)
                    if len(eval_loss):
                        print("Current Evaluation Loss: {}, {}, {}, {}".format(len(eval_loss), eval_loss.mean(), eval_loss.max(), eval_loss.min()))
                    break
