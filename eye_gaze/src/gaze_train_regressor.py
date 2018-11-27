""" Script to parse input arguments and start the training/evaluation process """

import os
import sys
import scipy.misc
import numpy as np
import pprint

from src.gaze_model_regressor import GazeRegressor
from src.gaze_evaluator_new import GazeEval
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("num_steps", 500000, "Epochs to train ")
flags.DEFINE_integer("decay_step", 50000, "Decay step of learning rate in steps")
flags.DEFINE_float("decay_rate", 0.5, "Decay rate of learning rate")
flags.DEFINE_float("gpu_frac", 0.9, "Gpu fraction")
flags.DEFINE_float("initial_learning_rate", 0.001, "Learing rate")

flags.DEFINE_string("train_data_path", "../data/syn_data_4_april_2018_norm/", "Directory name containing the dataset [data]")
flags.DEFINE_string("val_data_path", "../data/realMPII_train_val_test/", "Directory name containing the dataset [data]")

flags.DEFINE_boolean("evaluate", False, "Whether to evaluate a checkpoint or train?")
flags.DEFINE_string("log_eval_dir", "../logs/eval/", "Directory name to save the logs [logs]")
flags.DEFINE_string("loss_type", "cosine", "The loss function to train the model with")
flags.DEFINE_boolean("visualise", False, "Whether to visualise predictions?")
flags.DEFINE_string("visualise_dir", "../vis/", "Directory to store visualisations")

flags.DEFINE_string("checkpoint_dir", "../checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_file", "1.ckpt", "Name of the model checkpoint")
flags.DEFINE_string("log_dir", "../logs/gaze_train_mean_norm/", "Directory name to save the logs [logs]")
flags.DEFINE_boolean("load_chkpt", False, "True for loading saved checkpoint")

flags.DEFINE_integer("batch_size", 512, "The size of batch images [64]")
flags.DEFINE_integer("output_dim", 3, "Number of classes.")

flags.DEFINE_integer("channels", 1, "Number of channels in input image")
flags.DEFINE_integer("img_height", 35, "Height of Input Image")
flags.DEFINE_integer("img_width", 55, "Height of Input Image")

flags.DEFINE_integer("capacity", 512*10, "Capacity of input queue")
flags.DEFINE_integer("num_threads", 4, "Threads to employ for filling input queue")
flags.DEFINE_integer("min_after_dequeue", 512, "Minimum samples to remain after dequeue")

flags.DEFINE_integer("log_every", 50, "Frequency of logging for summary")
flags.DEFINE_integer("save_every", 500, "Save after steps")
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.log_eval_dir):
        os.makedirs(FLAGS.log_eval_dir)


    if not FLAGS.evaluate:
        source_task = GazeRegressor()
        source_task.train()
    else:
        eval_task = GazeEval()
        eval_task.eval()

if __name__ == '__main__':
    tf.app.run()
