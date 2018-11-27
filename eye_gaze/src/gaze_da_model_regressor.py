""" Script for domain adaptation """

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from dataio.data_reader import DAReader, CLSReader
from model.utils import collect_vars
from model.base_model import SimpleModel, MeanReLUModel, AdversaryModel

F = tf.app.flags.FLAGS
r2d = 180.0/3.14
syn_data_mean = 110.

class GazeDaRegressor():
    def __init__(self):
        # Initiliaze the data reader
        self.dataloader = DAReader()

        # Construct the model architecture to be learned
        self.build_model()

        self.source_data = self.dataloader.create_source_dataset()
        self.target_data = self.dataloader.create_target_dataset()

        self.source_iter = self.source_data.make_one_shot_iterator()
        self.target_iter = self.target_data.make_one_shot_iterator()

        self.eval_data = self.dataloader.create_val_dataset()
        self.eval_iter = self.eval_data.make_initializable_iterator()

    def get_loss(self):
        """ Attach the relevant loss depending on the type of GAN selected """

        if F.gan_type == "GAN":
            self.d_source_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.adversary_out_source, labels=tf.ones_like(self.adversary_out_source)))
            self.d_target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.adversary_out_target, labels=tf.zeros_like(self.adversary_out_target)))
            self.adversary_loss = self.d_target_loss + self.d_source_loss

            self.mapping_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.adversary_out_target, labels=tf.ones_like(self.adversary_out_target)))
        else:
            self.adversary_loss = tf.reduce_mean(self.adversary_out_target) - tf.reduce_mean(self.adversary_out_source)
            self.mapping_loss = -tf.reduce_mean(self.adversary_out_target)

            if F.gan_type == "WGAN_GP":
                epsilon = tf.random_uniform([F.batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated = epsilon * self.s + (1 - epsilon) * self.t
                adversary_model_interp = AdversaryModel(interpolated, scope='adversary', prob=[self.fc_prob, self.conv_prob], reuse=True)
                adversary_out_interp = adversary_model_interp.get_model()

                gradients = tf.gradients(adversary_out_interp, [interpolated, ], name="adversary_logits_interp")[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
                grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

                self.grad_penalty_summary = tf.summary.scalar("grad_penalty", grad_penalty)
                self.grad_norm_summary = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
                
                self.adversary_loss += F.gp_lambda * grad_penalty

        self.angle_loss = r2d*tf.acos(1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.target_labels,1), tf.nn.l2_normalize(self.target_out, 1), dim=1))

    def build_model(self):
        """ Loads the model from model/base_model.py """

        self.source_images, self.source_labels = self.dataloader.get_source_model_inputs()
        self.target_images, self.target_labels = self.dataloader.get_target_model_inputs()

        self.fc_prob = tf.placeholder_with_default(0.5, shape=())
        self.conv_prob = tf.placeholder_with_default(0.1, shape=())

        self.mean = tf.placeholder_with_default(syn_data_mean, shape=())
        source_model = MeanReLUModel(self.source_images, self.source_labels, F.output_dim, mean=self.mean, prob=0., scope='source_regressor')
        target_model = MeanReLUModel(self.target_images, self.target_labels, F.output_dim, mean=self.mean, prob=0., scope='target_regressor')

        self.source_out, self.s = source_model.get_model()
        self.target_out, self.t = target_model.get_model()

        # Construct the discriminator network
        adversary_model_source = AdversaryModel(self.s, scope='adversary', prob=[self.fc_prob, self.conv_prob])
        self.adversary_out_source = adversary_model_source.get_model()

        adversary_model_target = AdversaryModel(self.t, scope='adversary', prob=[self.fc_prob, self.conv_prob], reuse=True)
        self.adversary_out_target = adversary_model_target.get_model()

        self.get_loss()

    def train(self):
        """ Start the adaptatin process """

        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Adapting Source Network")

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Learning rate for the target model
        self.lr_t = tf.train.exponential_decay(
            learning_rate=F.initial_learning_rate,
            global_step=global_step,
            decay_steps=F.decay_step,
            decay_rate=F.decay_rate,
            staircase=True)

        # Learning rate for the source model
        self.lr_s = tf.train.exponential_decay(
            learning_rate=F.initial_learning_rate,
            global_step=global_step,
            decay_steps=F.decay_step,
            decay_rate=F.decay_rate,
            staircase=True)


        # variable collection for defining optimizer
        source_vars = collect_vars('source')
        target_vars = collect_vars('target')
        adversary_vars = collect_vars('adversary')

        # Now we can define the optimizer that takes on the learning rate
        if F.gan_type == "WGAN_GP":
            self.mapping_grad_update = tf.train.AdamOptimizer(learning_rate=self.lr_t, beta1=F.beta1, beta2=F.beta2).minimize(self.mapping_loss,
             global_step=global_step, var_list=target_vars.values())
            self.adversary_grad_update = tf.train.AdamOptimizer(learning_rate=self.lr_s, beta1=F.beta1, beta2=F.beta2).minimize(self.adversary_loss,
             global_step=global_step, var_list=adversary_vars.values())
        else:
            self.mapping_grad_update = tf.train.AdamOptimizer(learning_rate=self.lr_t).minimize(self.mapping_loss, global_step=global_step, 
                var_list=target_vars.values())
            self.adversary_grad_update = tf.train.AdamOptimizer(learning_rate=self.lr_s).minimize(self.adversary_loss, global_step=global_step,
                var_list=adversary_vars.values())


        self.map_loss_summary = tf.summary.scalar('mapping_loss', self.mapping_loss)
        self.adversary_loss_summary = tf.summary.scalar('adversary_loss', self.adversary_loss)
        self.summary_op = tf.summary.merge_all()

        # Saver object to save models
        self.source_saver = tf.train.Saver(max_to_keep=None, var_list=source_vars.values())
        self.target_saver = tf.train.Saver(max_to_keep=None, var_list=target_vars.values())
        self.all_saver = tf.train.Saver(max_to_keep=None)

        # Reload a saved checkpoint if required
        def restore_fn(sess):
            self.source_saver.restore(sess, F.source_checkpoint_dir + F.source_checkpoint_file)
            self.target_saver.restore(sess, F.target_checkpoint_dir + F.target_checkpoint_file)
            return

        self.source_handle_op = self.source_iter.string_handle()
        self.target_handle_op = self.target_iter.string_handle()
        self.eval_handle_op = self.eval_iter.string_handle()

        # Define your supervisor for running a managed session.
        # Point: if F.log_dir already contains a checkpoint; new training will start from there
        # Point: if load_chkpt True, load source & target models trained on syn
        if F.load_chkpt:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, init_fn=restore_fn, saver=self.all_saver)
        else:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, saver=self.all_saver)

        current_best_loss = 1000. #TODO: Read it from a file for multiple restarts
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.source_handle = sess.run(self.source_handle_op)
            self.target_handle = sess.run(self.target_handle_op)
            self.eval_handle = sess.run(self.eval_handle_op)

            for step in range(int(F.num_steps)):
                try:
                    if F.gan_type != "GAN":
                        _d_iters = 100 if step < 25 or np.mod(step, 500) == 0 else F.d_iter

                        for _ in range(_d_iters):
                            adversary_loss, _1, summaries, global_step_count = sess.run([self.adversary_loss, self.adversary_grad_update, self.summary_op, sv.global_step],
                                feed_dict={self.dataloader.source_split_handle: self.source_handle,
                                 self.dataloader.target_split_handle: self.target_handle,
                                 self.conv_prob: 0.1, self.fc_prob: 0.5})
                        
                        mapping_loss, _2, summaries, global_step_count = sess.run([self.mapping_loss, self.mapping_grad_update, self.summary_op, sv.global_step],
                            feed_dict={self.dataloader.source_split_handle: self.source_handle,
                             self.dataloader.target_split_handle: self.target_handle,
                             self.conv_prob: 0.1, self.fc_prob: 0.5})
                    else:
                        mapping_loss, adversary_loss, _1, _2, summaries, global_step_count = sess.run([self.mapping_loss, self.adversary_loss,
                        self.mapping_grad_update, self.adversary_grad_update, self.summary_op, sv.global_step],
                        feed_dict={self.dataloader.source_split_handle: self.source_handle,
                         self.dataloader.target_split_handle: self.target_handle,
                         self.conv_prob: 0.1, self.fc_prob: 0.5})
                    logging.info("Step: {}/{}, Global Step: {}, mapping_loss: {}, adversary_loss: {}".format(step,
                     F.num_steps, global_step_count, mapping_loss, adversary_loss))
                except:
                    logging.info("Smaller batch size error,.. proceeding to next batch size")

                # print current performance on validation set to monitor training
                if step % F.check_every==1:
                    logging.info("Evaluating target model on data")
                    sess.run(self.eval_iter.initializer)
                    eval_loss = []
                    # i = 4
                    while True:
                        try:
                            loss = sess.run([self.angle_loss], feed_dict={self.dataloader.source_split_handle: self.source_handle,
                             self.dataloader.target_split_handle: self.eval_handle,
                             self.conv_prob: 0.0, self.fc_prob: 0.0})
                            eval_loss.append(loss)
                        except:
                            if len(eval_loss):
                                eval_loss = np.array(eval_loss)
                                logging.info("Current Evaluation Loss at step({}): {}, Mean Angle Loss: {}, {}, {}".format(step, len(eval_loss), eval_loss.mean(), eval_loss.max(), eval_loss.min()))
                                break
                            else:
                                print("No Evaluation Loss as eval_loss array empty")
                                break
