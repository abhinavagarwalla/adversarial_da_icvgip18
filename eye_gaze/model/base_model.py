""" All model architecture for use in the project """

from ops import *
import tensorflow as tf
import sys


class MeanReLUModel():
    """ Generator/Regressor Version A """
    def __init__(self, images, labels, output_dim=None, mean=None, prob=0.5, scope=None):
        """
        Args:
            images: training images
            labels: corresponding labels
            output_dim: output dimension of the network [#classes]
            mean: mean of training data samples to be subtracted
            prob: dropout probability
            scope: model scope

        """
        self.images = images
        self.labels = labels
        self.output_dim = output_dim
        self.scope = scope
        self.mean = mean
        self.prob = prob

    def adaptation(self, h2, h4,l2):
        """
        Args:
            h2: 2nd hidden layer
            h4: 4th hidden layer
            l2: 2nd fully connected layer

        Returns:
            Concats all features together to be judged by discriminator
        """
        h2_shape = h2.get_shape().as_list()
        h4_reshaped = tf.image.resize_bilinear(h4, [32, 60])
        h2_reshaped = tf.image.resize_bilinear(h2, [32, 60])
        l2 = tf.reshape(l2, [-1, 32, 60, 5])
        a1 = tf.concat([h2_reshaped, h4_reshaped, l2], axis=3)
        return a1

    def get_model(self):
        """ Builds the model architecture
        Returns:
            out: regressor output [gaze angles]
            to_adapt: features to be adapted
        """
        with tf.variable_scope(self.scope):
            input0 = mean_normalize(self.images, self.mean)
            h0 = tf.nn.relu(conv2d(input0, 32, 3, 3, 1, 1, name='h0_conv'))
            h1 = tf.nn.relu(conv2d(h0, 32, 3, 3, 1, 1, name='h1_conv'))
            h2 = tf.nn.relu(conv2d(h1, 64, 3, 3, 1, 1, name='h2_conv'))
            
            p1 = tf.layers.max_pooling2d(h2, 3, 2, name='pool1')
            h3 = tf.nn.relu(conv2d(p1, 80, 3, 3, 1, 1, name='h3_conv'))
            h4 = tf.nn.relu(conv2d(h3, 192, 3, 3, 1, 1, name='h4_conv'))
            p2 = tf.layers.max_pooling2d(h4, 2, 2, name='pool2')
            l1 = tf.contrib.layers.flatten(p2)

            l2 = tf.nn.relu(linear(l1, 9600, scope="Linear9600"))
            l2 = tf.layers.dropout(inputs=l2, rate=self.prob, name='l2_dropout')
            l3 = tf.nn.relu(linear(l2, 1000, scope="Linear1000"))
            l3 = tf.layers.dropout(inputs=l3, rate=self.prob, name='l3_dropout')
            out = linear(l3, self.output_dim)

            to_adapt = self.adaptation(h2, h4, l2)
            return out, to_adapt


class SimpleModel():
    """ Generator/Reressor Version B """
    def __init__(self, images, labels, output_dim=None, scope=None):
        """
        Args:
            images: training images
            labels: corresponding labels
            output_dim: output dimension of the network [#classes]
            scope: model scope

        """
        self.images = images
        self.labels = labels
        self.output_dim = output_dim
        self.scope = scope

    def adaptation(self, h2, h4,l2):
        """
        Args:
            h2: 2nd hidden layer
            h4: 4th hidden layer
            l2: 2nd fully connected layer

        Returns:
            Concats all features together to be judged by discriminator
        """
        h2_shape = h2.get_shape().as_list()
        h4_reshaped = tf.image.resize_bilinear(h4, [32, 60])
        h2_reshaped = tf.image.resize_bilinear(h2, [32, 60])
        l2 = tf.reshape(l2, [-1, 32, 60, 5])
        a1 = tf.concat([h2_reshaped, h4_reshaped, l2], axis=3)
        return a1

    def get_model(self):
        """ Builds the model architecture
        Returns:
            out: regressor output [gaze angles]
            to_adapt: features to be adapted
        """
        with tf.variable_scope(self.scope):
            input0 = normalize(self.images)
            h0 = lrelu(conv2d(input0, 32, 3, 3, 1, 1, name='h0_conv'), 0.2)
            h1 = lrelu(conv2d(h0, 32, 3, 3, 1, 1, name='h1_conv'), 0.2)
            h2 = lrelu(conv2d(h1, 64, 3, 3, 1, 1, name='h2_conv'), 0.2)
            
            p1 = tf.layers.max_pooling2d(h2, 3, 2, name='pool1')
            h3 = lrelu(conv2d(p1, 80, 3, 3, 1, 1, name='h3_conv'), 0.2)
            h4 = lrelu(conv2d(h3, 192, 3, 3, 1, 1, name='h4_conv'), 0.2)
            p2 = tf.layers.max_pooling2d(h4, 2, 2, name='pool2')
            l1 = tf.contrib.layers.flatten(p2)

            l2 = lrelu(linear(l1, 9600, scope="Linear9600"), 0.2)
            l3 = lrelu(linear(l2, 1000, scope="Linear1000"), 0.2)
            out = linear(l3, self.output_dim)

            to_adapt = self.adaptation(h2, h4, l2)
            return out, to_adapt


class AdversaryModel():
    """ 3D Discriminator """
    def __init__(self, input_vec=None, scope=None, prob=None, reuse=False):
        """
        Args:
            input_vec: Input feature vectors from generator
            scope: model scope
            prob: dropout probability for both conv and fc layers
            reuse: whether to reuse layer values
        """
        self.scope = scope
        self.input = input_vec
        if prob is not None:
            self.fc_prob = prob[0]
            self.conv_prob = prob[1]
        self.reuse = reuse

    def get_model(self):
        """ Builds the discriminator architecture
        Returns:
            out: classifier output of discriminator [real or fake]
        """
        with tf.variable_scope(self.scope):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            shape = self.input.get_shape().as_list()
            self.input = tf.reshape(self.input, [-1,shape[3], shape[1], shape[2], 1])
            h1 = lrelu(conv3d(input_=self.input, out_channels=16, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h1_conv3d'), 0.2)
            h1 = tf.layers.dropout(inputs=h1, rate=self.conv_prob, name='conv1_drop')

            h2 = lrelu((conv3d(input_=h1, out_channels=32, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h2_conv3d')), 0.2)
            h2 = tf.layers.dropout(inputs=h2, rate=self.conv_prob, name='conv2_drop')
            h3 = lrelu((conv3d(input_=h2, out_channels=64, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h3_conv3d')), 0.2)
            h3 = tf.layers.dropout(inputs=h3, rate=self.conv_prob, name='conv3_drop')
            h4 = lrelu((conv3d(input_=h3, out_channels=128, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h4_conv3d')), 0.2)
            h4 = tf.layers.dropout(inputs=h4, rate=self.conv_prob, name='conv4_drop')
         
            l1 = tf.contrib.layers.flatten(h4)
            l1 = tf.layers.dropout(inputs=l1, rate=self.fc_prob, name="adv_dropout")
            out = linear(l1, 1, scope="final_decision")
            return out
