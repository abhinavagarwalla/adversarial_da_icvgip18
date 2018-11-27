""" Script to visualize the model output """

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from dataio.data_reader import CLSReader
from model.utils import collect_vars
from model.base_model import SimpleModel, AdversaryModel
import pprint, os
from PIL import Image
import cv2

F = tf.app.flags.FLAGS
r2d = 180.0/3.14
save_dir = "/home/avisek/domain_adaptation/eye_gaze/checks/images/"


class GazeTestRegressor():
    def __init__(self):
        self.dataloader = CLSReader()

        self.build_model()

        self.test_data = self.dataloader.create_test_dataset()
        self.test_iter = self.test_data.make_one_shot_iterator()

    def get_loss(self):
        """ computes all the losses """
        self.lbl = tf.nn.l2_normalize(self.labels, 1)
        self.source_out = tf.nn.l2_normalize(self.source_out, 1)
        self.source_loss = r2d*tf.acos(1-tf.losses.cosine_distance(self.lbl, self.source_out, dim=1))

        self.target_out = tf.nn.l2_normalize(self.target_out, 1)
        self.target_loss = r2d*tf.acos(1-tf.losses.cosine_distance(self.lbl, self.target_out, dim=1))

    def build_model(self):
        """ build source and target models """
        self.images, self.labels = self.dataloader.get_model_inputs()

        source_model = SimpleModel(self.images, self.labels, F.output_dim, scope='source_regressor')
        target_model = SimpleModel(self.images, self.labels, F.output_dim, scope='target_regressor')
        
        self.source_out, _ = source_model.get_model()
        self.target_out, _ = target_model.get_model()

        self.get_loss()

    def test(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Visualizin Source-Target Network")

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # variable collection
        source_vars = collect_vars('source')
        target_vars = collect_vars('target')

        self.source_saver = tf.train.Saver(max_to_keep=None, var_list=source_vars.values())
        self.target_saver = tf.train.Saver(max_to_keep=None, var_list=target_vars.values())

        def restore_fn(sess):
            self.source_saver.restore(sess, F.source_checkpoint_dir + F.source_checkpoint_file)
            self.target_saver.restore(sess, F.target_checkpoint_dir + F.target_checkpoint_file)
            return

        self.test_handle_op = self.test_iter.string_handle()

        # Define your supervisor for running a managed session.
        if F.load_chkpt:
            sv = tf.train.Supervisor(logdir=F.log_eval_dir, summary_op=None, init_fn=restore_fn, saver=None)
        else:
            sv = tf.train.Supervisor(logdir=F.log_eval_dir, summary_op=None, init_fn=None, saver=None)

        current_best_loss = 1000. #TODO: Read it from a file for multiple restarts
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.test_handle = sess.run(self.test_handle_op)

            eval_loss = []
            while True:
                # try:
                loss_source, loss_target, img, lbl, source_pred, target_pred = sess.run([self.source_loss, self.target_loss,
                 self.images, self.labels, self.source_out, self.target_out],
                    feed_dict={self.dataloader.split_handle: self.test_handle})
                logging.info("Batch-Loss Source: {}, Target: {}".format(loss_source, loss_target))
                
                img = img[0].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                eye_c = [img.shape[1]/2, img.shape[0]/2]
                print(img.shape, eye_c, source_pred, target_pred)
                cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(source_pred[0][:2])*80).astype(int)), (255,0,0), 2)
                cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(target_pred[0][:2])*80).astype(int)), (0,255,0), 2)
                cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(lbl[0][:2])*80).astype(int)), (0,0,255), 2)
                cv2.imwrite(save_dir + np.array_str(lbl[0]) + '_' + '(' + str(loss_source) + ',' + str(loss_target) + '.jpg', img)


flags = tf.app.flags
flags.DEFINE_integer("num_steps", 500000, "Epochs to train ")
flags.DEFINE_float("gpu_frac", 0.25, "Gpu fraction")

flags.DEFINE_string("test_data_path", "../data/realMPII_train_val_test/", "Directory name containing the dataset [data]")
flags.DEFINE_string("log_eval_dir", "../logs/test/", "Directory name to save the logs [logs]")

flags.DEFINE_string("source_checkpoint_dir", "../logs/latest_model_15/source/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("source_checkpoint_file", "model.ckpt", "Name of the model checkpoint")
flags.DEFINE_string("target_checkpoint_dir", "../logs/da/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("target_checkpoint_file", "model.ckpt-4116", "Name of the model checkpoint")

flags.DEFINE_boolean("load_chkpt", True, "True for loading saved checkpoint")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("output_dim", 3, "Number of classes.")

flags.DEFINE_integer("channels", 1, "Number of channels in input image")
flags.DEFINE_integer("img_height", 35, "Height of Input Image")
flags.DEFINE_integer("img_width", 55, "Height of Input Image")

flags.DEFINE_integer("capacity", 1*10, "Capacity of input queue")
flags.DEFINE_integer("num_threads", 4, "Threads to employ for filling input queue")
flags.DEFINE_integer("min_after_dequeue", 1, "Minimum samples to remain after dequeue")

flags.DEFINE_integer("log_every", 500, "Frequency of logging for summary")
flags.DEFINE_integer("save_every", 5000, "Save after steps")
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.log_eval_dir):
        os.makedirs(FLAGS.log_eval_dir)

    task = GazeTestRegressor()
    task.test()

if __name__ == '__main__':
    tf.app.run()