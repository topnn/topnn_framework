from transformers import Transformer
import tensorflow as tf
import os

class TFSaver(Transformer):
    def __init__(self, output_filename="train.tfrecords"):
            if not os.path.exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename))

            self._writer = tf.python_io.TFRecordWriter(output_filename)
            self._output_filename = output_filename


    def write(self, point, gt_label):
        feature = {
            'vector/point': tf.train.Feature(float_list=tf.train.FloatList(value=point)),
            'vector/ground_truth': tf.train.Feature(float_list=tf.train.FloatList(value= [gt_label]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example.SerializeToString())

    def transform(self, content=None):
        ll = len(content[0])
        for i in range(ll):
            self.write(content[0][i], content[1][i])
        self._writer.close()


