""" Script to check if tfrecords are read correctly and get basic one-off statistics about data """

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from os.path import expanduser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# data_path from which tfrecord will be read
data_path = expanduser("~") + '/domain_adaptation/data/syn_data_4_april_2018_norm/train_003_009.tfrecords'
split = 'test'

with tf.Session() as sess:
    feature = {split + '/image': tf.FixedLenFeature([], tf.string),
                split + '/label': tf.FixedLenFeature([], tf.string)}

     # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features[split + '/image'], tf.float32)
    image = tf.reshape(image, [35, 55, 1])

    label = tf.decode_raw(features[split + '/label'], tf.float32)
    label = tf.reshape(label, [3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=512, capacity=512*4, num_threads=16, min_after_dequeue=512)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    x_mean, x_std = [], []
    try:
        for batch_index in range(500):
            img, lbl = sess.run([images, labels])
            img = img.astype(np.uint8)
            # print(np.max(img), np.min(img))
            # print("Images, labels shape size: ", img.shape, lbl.shape)
            #in case of mean image, np.mean(img, axis=0)
            # print(np.mean(img))
            x_mean.append(np.mean(img))
            x_std.append(np.std(img))
            # exit()
            # print("Labels: ",  lbl)
            # for j in range(6):
               # im = Image.fromarray(img[j].reshape(35, 55))
               # im.save(np.array_str(lbl[j]) + '_' + str(j) + '.jpg')
    except:
        print(np.mean(x_mean), np.mean(x_std))
    
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
