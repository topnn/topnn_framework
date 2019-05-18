import glob
import os
import tensorflow as tf
import numpy as np

vec_size = 2

feature = {
    'vector/point': tf.FixedLenFeature(shape=[vec_size], dtype=tf.float32),
    'vector/ground_truth': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
}

class DataLoader(object):
    def __init__(self,
                 shard_path,
                 batch_size=100,
                 num_parallel_calls = 1,
                 prefetch_buffer_size = 500):

        dataset_files = self._parse_input(shard_path)
        self._check_filenames(dataset_files)

        if len(dataset_files) == 0:
            raise IOError('Error: dataset files do not exist.')
        _, ext = os.path.splitext(dataset_files[0])
        if ext == '.tfrecords':
            self._dataset = tf.data.TFRecordDataset(dataset_files,
                                                    num_parallel_reads=num_parallel_calls)

        self._dataset = self._dataset.map(parse,
                                          num_parallel_calls=num_parallel_calls)

        self._dataset = self._dataset.batch(batch_size=batch_size)
        self._dataset = self._dataset.prefetch(buffer_size=prefetch_buffer_size)
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    def initialize(self, sess):
        sess.run(self._iterator.initializer)

    def _parse_input(self, shard_path):
        if isinstance(shard_path, str):
            if os.path.isdir(shard_path):
                shard_files = glob.glob(os.path.join(shard_path, '*.tfrecords'))
            else:
                shard_files = glob.glob(shard_path)
        elif isinstance(shard_path, list):
            shard_files = []
            for f in shard_path:
                if os.path.isdir(f):
                    new_path = os.path.join(f, '*.tfrecords')
                    shard_files += self._parse_input(new_path)
                else:
                    shard_files.append(f)
        else:
            raise FileNotFoundError(shard_path)
        return sorted(shard_files)

    @staticmethod
    def _check_filenames(filenames):
        for f in filenames:
            if not os.path.exists(f):
                raise FileNotFoundError(f)

    def load_batch(self, sess):
        batch = sess.run(self._next_batch)
        return batch


def parse(example):
    parsed_example = tf.parse_single_example(example, features=feature)
    return  parsed_example


if __name__ == '__main__':
    import time
    shards = '/home/topology_of_dl/'
    batch_size = 16
    num_parallel_calls = 5
    prefetch_buffer_size = 100
    loopruns = 5
    data_loader = DataLoader(shards,
                             batch_size=batch_size,
                             num_parallel_calls=num_parallel_calls)

    with tf.Session() as sess:
        data_loader.initialize(sess)
        total_time = 0
        for i in range(loopruns):
            start_time = time.time()
            batch = data_loader.load_batch(sess)
            total_time += (time.time() - start_time)

        print("FPS: {0:.2f}".format(batch_size*loopruns/total_time))