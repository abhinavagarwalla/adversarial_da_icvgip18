""" Script for converting MPIIGaze dataset into TFRecords parallely"""

from __future__ import print_function
import glob
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tqdm import tqdm
import numpy as np

from gaze_cropper import process_json_list
from gaze_converter import _float_feature, _bytes_feature
from os.path import expanduser
from pathos.multiprocessing import ProcessPool as Pool
from scipy.io import loadmat
from scipy.misc import imresize

import tensorflow as tf

def load_image_label(addr):
    """ Read the gaze vectors from MAT file """

    data_arr = loadmat(addr)
    data = data_arr['data'][0][0][1][0][0]
    imgs, look_vecs = [], []
    for idx in range(data[0].shape[0]):
        look_vecs.append(data[0][idx,:])
        imgs.append(imresize(data[1][idx][1:,3:-2], (35, 55)))

    return np.array(imgs).astype(np.float32), np.array(look_vecs).astype(np.float32)

def write_record(train_filename, addrs, split):
    """ Load eye images, gazee vectors and write to TFRecord file """
    
    writer = tf.python_io.TFRecordWriter(train_filename)
    for idx in tqdm(range(len(addrs))):
        try:
            imgs, labels = load_image_label(addrs[idx])

            for i in range(len(imgs)):
                # Create a feature
                feature = {split + '/label': _bytes_feature(tf.compat.as_bytes(labels[i].tostring())),
                           split + '/image': _bytes_feature(tf.compat.as_bytes(imgs[i].tostring()))}
                
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
        except:
            print("Couldn't write to tf, exception raised..moving to next image")

    writer.close()

def write_data(img_data_path):
    """ Divide dataset into training, validation and test sets and store them as TFRecords """

    patients = os.listdir(img_data_path)
    train_addrs = [glob.glob(img_data_path + i + '/*') for i in patients[0:int(0.75*len(patients))]]
    val_addrs = [glob.glob(img_data_path + i + '/*') for i in patients[int(0.75*len(patients)):int(0.90*len(patients))]]
    test_addrs = [glob.glob(img_data_path + i + '/*') for i in patients[int(0.9*len(patients)):]]

    n_shards = 8
    train_filenames = [expanduser("~") + '/domain_adaptation/eye_gaze/data/realMPII/{}_{:0>3}_{:0>3}.tfrecords'.format('train', i, n_shards-1) for i in range(n_shards)]
    val_filenames = [expanduser("~") + '/domain_adaptation/eye_gaze/data/realMPII/{}_{:0>3}_{:0>3}.tfrecords'.format('val', i, n_shards-1) for i in range(n_shards)]
    test_filenames = [expanduser("~") + '/domain_adaptation/eye_gaze/data/realMPII/{}_{:0>3}_{:0>3}.tfrecords'.format('test', i, n_shards-1) for i in range(n_shards)]

    #write_record(train_filenames[0], train_addrs[0], 'train')
    p = Pool(n_shards)
    p.map(write_record, train_filenames, train_addrs, ['train' for i in range(n_shards)])
    p.map(write_record, val_filenames, val_addrs, ['val' for i in range(n_shards/2)])
    p.map(write_record, test_filenames, test_addrs, ['test' for i in range(n_shards/2)])
    sys.stdout.flush()

if __name__=="__main__":
    home = expanduser("~")
    img_data_path = home + '/domain_adaptation_datasets/MPIIGaze/Data/Normalized/'
    write_data(img_data_path)