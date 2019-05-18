from transformers import Transformer
import tensorflow as tf
import numpy as np

class TFReader(Transformer):
    def __init__(self, content_name ='grid', input_filename="train.tfrecords", mode= 'normal'):
        self._input_filename = input_filename
        self.content_name = content_name
        self.mode = mode

    def write(self, point, gt_label):
        feature = {
            'vector/point': tf.train.Feature(float_list=tf.train.FloatList(value=point)),
            'vector/ground_truth': tf.train.Feature(float_list=tf.train.FloatList(value= [gt_label]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example.SerializeToString())

    def transform(self, content=None):
        """Read tfrecords database
        """

        def get_sample(dataset):
            """Extract ground truth and point value. """

            items = tf.train.Example.FromString(dataset)
            point_as_list = items.features.feature.get('vector/point').float_list.value
            samples = [point_as_list.pop() for i in range(len(point_as_list))]
            return samples

        def get_label(dataset):

            items = tf.train.Example.FromString(dataset)
            labels = items.features.feature.get('vector/ground_truth').float_list.value

            return labels

        iter = tf.python_io.tf_record_iterator(self._input_filename)
        samples = [get_sample(item) for item in iter]

        iter = tf.python_io.tf_record_iterator(self._input_filename)
        labels = [get_label(item) for item in iter]

        if self.mode == 'test':

            dic = {self.content_name: {'samples' : np.array(samples),
                                       'labels' : labels,
                                       'label_names' : ['cat'] * len(labels)}}
        else:
            dic = {self.content_name: {'samples' : samples,
                                       'labels' : labels,
                                       'label_names' : ['cat'] * len(labels)}}

        if content != None:
            content.update(dic)
        else:
            content = dic

        return content

