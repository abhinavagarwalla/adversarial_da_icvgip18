""" Main file for data ingestion into the network """

import tensorflow as tf
from glob import glob

F = tf.app.flags.FLAGS

class CLSReader():
    """ Data reader for pre-training the generator [source network] """
    def __init__(self):
        self.split_handle = tf.placeholder(tf.string, shape=[])
        self.train_split, self.val_split, self.test_split = None, None, None
        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))
        self.next_element = self.iterator.get_next()

        self.split = None

    def create_training_dataset(self):
        """ reads training tfrecords """

        self.split = 'train'
        filenames = glob(F.train_data_path + 'train*.tfrecords')
        print(filenames)
        self.train_split = self.create_dataset(filenames)
        return self.train_split

    def create_validation_dataset(self):
        """ reads validation tfrecords
        currently allows only 1 iteration through the data, must be initialised again for another iteration
        """

        self.split = 'val'
        filenames = glob(F.val_data_path + 'val*.tfrecords')
        self.val_split = self.create_dataset(filenames, 1)
        return self.val_split

    def create_test_dataset(self):
        """ reads test tfrecords
        currently allows only 1 iteration through the data, must be initialised again for another iteration
        """

        self.split = 'test'
        filenames = glob(F.test_data_path + 'test*.tfrecords')
        print(filenames)
        self.test_split = self.create_dataset(filenames, 1)
        return self.test_split

    def create_dataset(self, filenames, num_epochs=None):
        """
        Args:
            filenames: TFRecords files to be loaded
            num_epochs: number of epochs allowed, None ~ no limit
        """

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_records, num_threads=F.num_threads, output_buffer_size=F.capacity)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(F.batch_size)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        return dataset

    def parse_records(self, serialized_example):
        """ fetch a single training data point from the TFRecords """
        feature = {self.split + '/image': tf.FixedLenFeature([], tf.string),
                self.split + '/label': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[self.split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[self.split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        return image, label

    def get_model_inputs(self):
        """
        Returns:
            next_element: data iterator
        """
        return self.next_element

class DAReader():
    """ Data Reader for training target network and discriminator """
    def __init__(self):
        self.source_split_handle = tf.placeholder(tf.string, shape=[])
        self.target_split_handle = tf.placeholder(tf.string, shape=[])
        self.source_split, self.target_split, self.test_split = None, None, None

        self.source_iterator = tf.contrib.data.Iterator.from_string_handle(self.source_split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))
        self.target_iterator = tf.contrib.data.Iterator.from_string_handle(self.target_split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))

        self.next_source_element = self.source_iterator.get_next()
        self.next_target_element = self.target_iterator.get_next()

        self.split = None

    def create_source_dataset(self):
        """ reads source training tfrecords """

        self.split = 'train'
        filenames = glob(F.source_data_path + 'train*.tfrecords')
        print(filenames)
        self.source_split = self.create_dataset(filenames)
        return self.source_split

    def create_target_dataset(self):
        """ reads target training tfrecords """

        self.split = 'train'
        filenames = glob(F.target_data_path + 'train*.tfrecords')
        print(filenames)
        self.target_split = self.create_dataset(filenames)
        return self.target_split

    def create_val_dataset(self):
        """ reads validation tfrecords """

        self.split = 'val'
        filenames = glob(F.test_data_path + 'val*.tfrecords')
        print(filenames)
        self.test_split = self.create_dataset(filenames)
        return self.test_split

    def create_dataset(self, filenames, num_epochs=None):
        """
        Args:
            filenames: TFRecords files to be loaded
            num_epochs: number of epochs allowed, None ~ no limit
        """

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_records, num_threads=F.num_threads, output_buffer_size=F.capacity)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(F.batch_size)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        return dataset

    def parse_records(self, serialized_example):
        """ fetch a single training data point from the TFRecords """
        
        feature = {self.split + '/image': tf.FixedLenFeature([], tf.string),
                self.split + '/label': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[self.split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[self.split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        return image, label

    def get_source_model_inputs(self):
        """
        Returns:
            next_source_element: data iterator for source data
        """
        return self.next_source_element

    def get_target_model_inputs(self):
        """
        Returns:
            next_target_element: data iterator for target data
        """
        return self.next_target_element