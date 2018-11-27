""" Script to convert processed images into TFRecords """

from __future__ import print_function
from random import shuffle, random
import glob
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.misc import imread, imresize
from tqdm import tqdm
import numpy as np
from gaze_cropper import process_json_list
from os.path import expanduser
from pathos.multiprocessing import ProcessPool as Pool

import tensorflow as tf
import json

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img = imread(addr)
    return img.astype(np.float32)

def load_label(addr):
    with open(addr) as json_f:
        j = json.loads(json_f.read())
        j = map(float, j["eye_details"]["look_vec"].encode('utf-8')[1:-1].split(','))
        look_vec = np.array(j[:3]).astype(np.float32)
        look_vec[1] = -look_vec[1]
    return look_vec

def write_record(train_filename, addrs, labels, split):
    writer = tf.python_io.TFRecordWriter(train_filename)

    for idx in tqdm(range(len(addrs))):
        # print(idx)
        try:
            img = load_image(addrs[idx])
            label = load_label(labels[idx])
            
            # Create a feature
            feature = {split + '/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                       split + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

            if split == "train":
                if random() > 0.5:
                    img = np.fliplr(img)
                    label[0] = -label[0]
                    # Create a feature
                    feature = {split + '/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                               split + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    
                    # Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
        except:
            print("Couldn't write to tf, exception raised..moving to next image")

    writer.close()

def write_para_record(addrs, labels, split='train', n_shards=10):
    train_filenames = [expanduser("~") + '/domain_adaptation/data/syn_data_4_april_2018_norm/{}_{:0>3}_{:0>3}.tfrecords'.format(split, i, n_shards-1) for i in range(n_shards)]
    addrs_split, labels_split = np.array_split(np.array(addrs), n_shards), np.array_split(np.array(labels), n_shards)
    # print(type(addrs_split), len(addrs_split), addrs_split[0].shape)
    p = Pool(n_shards)
    p.map(write_record, train_filenames, addrs_split, labels_split, [split for i in range(n_shards)])
    # write_record(train_filenames[0], addrs_split, labels_split, split)
    sys.stdout.flush()

def write_data(syn_data_path, syn_labels_path):
    shuffle_data = True

    addrs = glob.glob(syn_data_path)
    labels = [ad.replace('imgs_cropped', 'imgs').replace('_cropped.png', '.json') for ad in addrs]

    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 80% train, 10% validation, and 10% test
    train_addrs = addrs[0:int(0.8*len(addrs))]
    train_labels = labels[0:int(0.8*len(labels))]
    val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
    val_labels = labels[int(0.8*len(addrs)):int(0.9*len(addrs))]
    test_addrs = addrs[int(0.9*len(addrs)):]
    test_labels = labels[int(0.9*len(labels)):]

    write_para_record(train_addrs, train_labels, split='train')
    write_para_record(val_addrs, val_labels, split='val', n_shards=5)
    write_para_record(test_addrs, test_labels, split='test', n_shards=5)

if __name__=="__main__":
    home = expanduser("~")
    syn_data_path = home + '/domain_adaptation_datasets/eye_gaze_4_april_2018/imgs_cropped/*.png'
    syn_labels_path = home + '/domain_adaptation_datasets/eye_gaze_4_april_2018/imgs/'
    write_data(syn_data_path, syn_labels_path)
