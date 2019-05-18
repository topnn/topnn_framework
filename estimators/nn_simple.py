from __future__ import print_function
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
import os
import time
import glob
import argparse
from transformers.tf_reader import TFReader
from transformers.grid_generator import GridGenerator
from transformers.visualizer_2d import Visualizer2D
from importlib import import_module
import pipelines
from transformers.pickle_tensors_values_saver import PickleTensorValuesSaver
from transformers.excel_tensor_values_saver import ExcelTensorValuesSaver
from transformers.visualizer_2d import Visualizer2D
from transformers.pickle_reader import PickleReader
from transformers.tf_reader import TFReader
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import traceback

import numbers

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn


def custom(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.

  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  with ops.name_scope(name, "Custom", [features, alpha]) as name:
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.to_float(features)
    alpha = ops.convert_to_tensor(0, dtype=features.dtype, name="alpha")
    return tf.identity(math_ops.minimum(math_ops.maximum(features, 0, name='CustomAmax'), 1), name=name)



class NeuralNetworkSimple(ABC):

    @staticmethod
    def one_hot_encode(x, n_classes):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
         """
        return np.eye(n_classes)[x]


    def neural_net(self, x, model_type, activation_type):

        ACTIVATION_NAMES = ['LeakyRelu', 'Relu', 'Tanh', 'Custom']
        ACTIVATIONS = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.tanh, custom]


        MODELS = [ '4_by_50',
                   '2_by_15',
                   '3_by_15',
                  '10_by_5',
                  '10_by_6',
                  '10_by_10',
                  '10_by_15',
                  '8_by_15',
                  '4_by_15',
                  '10_by_25',
                  '10_by_25',
                  '10_by_50',
                  '3_by_10_2_by_1_3_by_10',
                  '3_by_10_3_by_1_3_by_10',
                  '3_by_25_2_by_1_3_by_25',
                  '3_by_25_3_by_1_3_by_25',
                  '5_by_15',
                  '6_by_15',
                  '7_by_15']

        MODEL_PARAMS = [(4, 50),
                        (2, 15),
                        (3, 15),
                        (10, 5),
                         (10, 6),
                         (10, 10),
                         (10, 15),
                         (8,15),
                         (4,15),
                         (10, 25),
                         (10, 25),
                         (10, 50),
                         (3,10, 2),
                         (3, 10, 3),
                         (3, 25, 2),
                         (3, 25, 3),
                         (5, 15),
                         (6, 15),
                         (7, 15)
                        ]


        i = ACTIVATION_NAMES.index(activation_type)
        activation = ACTIVATIONS[i]

        j = MODELS.index(model_type)
        model_params = MODEL_PARAMS[j]

        width = model_params[1]
        length = model_params[0]
        num_classes = 2

        layers = []
        layers.append(activation(tf.layers.dense(x, width)))
        for i in range(0, 20):
            layers.append(activation(tf.layers.dense(layers[i], width)))

        # identity to make the name look nice
        out_layer = tf.identity(tf.layers.dense(layers[length-2], num_classes))

        print("Names of the layer in a network")
        print("input:", x)
        for j in range(length-1):
            print("hidden layer:", layers[j])
        print("outer most layer before soft max:", out_layer)

        return out_layer

    def __init__(self,
                 model_path: str = 'models_folder',
                 num_input = 2,
                 **fit_params):

        self.model_type = fit_params.get('model', '10_by_5')
        self.activation_type = fit_params.get('activation_type', 'LeakyRelu')
        self.activation_name = self.activation_type

        self._model_path = model_path
        self.learning_rate = fit_params.get('learning_rate', 0.001)
        self.restore_meta_graph = fit_params.get('restore_meta_graph', False)
        self.freeze_pretrained = fit_params.get('freeze_pretrained', False)
        self.verbose = fit_params.get('verbose', False)
        output_folder = fit_params.get('output', './output')

        # directory where we are going to save training logs
        logs_path = os.path.join(output_folder, 'logs')
        os.makedirs(logs_path, exist_ok=True)
        self.logs_path = logs_path

        # prefix where we are going to snapshots of our save trained models
        os.makedirs(os.path.join(output_folder, 'snapshots'), exist_ok=True)
        model_snapshot_save_path = os.path.join(output_folder, 'snapshots','snapshot')
        self.model_snapshot_save_path = model_snapshot_save_path

        num_input = num_input
        num_classes = 2
        self.X = tf.placeholder("float", [None, num_input])
        self.Y = tf.placeholder("float", [None, num_classes])


        self.logits = self.neural_net(self.X, self.model_type, self.activation_type)
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        self.step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, self.step, 2500, 0.5)
        # rate = tf.train.exponential_decay(self.learning_rate, self.step, 4000, 0.5)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=rate)

        self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.step)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss_op)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", self.accuracy)
        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()
        self._pretrained_model = fit_params.get('pretrained_model')

        self._saver = tf.train.Saver(save_relative_paths=True, max_to_keep=1)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._model_loaded = False

    def _load_model_decorator(func):
        """
        Decorator function for loading the TF model. Used before making predictions.
        """

        def wrapper(self, *arg, **kw):
            if not self._model_loaded:
                self._load_model(os.path.join(self._model_path))

                self._model_loaded = True

            return func(self, *arg, **kw)

        return wrapper

    def _predict_one(self, input_vec):
        feed_dict = {self._in: input_vec}
        try:
            out_all = self._sess.run(self._pred, feed_dict)
        except ValueError as e:
            print(str(e))
            raise ValueError(
                "Something went wrong")
        out = out_all
        return out

    @_load_model_decorator
    def transform(self, input_vec, model_path = None):
        """ Return prediction and data"""

        return_dic = False
        if isinstance(input_vec, dict):
            contents = input_vec
            input_vec = input_vec['test_dataset']['samples']
            return_dic = True

        if model_path:
            self._model_path = model_path

        if type(input_vec) is np.ndarray and input_vec.shape[0] == 1:
            if return_dic:
                contents.update({'nn_predictions_on_test_dataset':{'labels': self._predict_one(input_vec),
                    'samples': input_vec,
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}})
                return contents

            return {'labels': self._predict_one(input_vec),
                    'samples': input_vec,
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}

        elif type(input_vec) is np.ndarray:

            if return_dic:
                contents.update({'nn_predictions_on_test_dataset':{'labels': [self._predict_one(input_vec[np.newaxis, i, :]) for i in range(input_vec.shape[0])],
                    'samples': [input_vec[i, :] for i in range(input_vec.shape[0])],
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}})
                return contents

            return {'labels': [self._predict_one(input_vec[np.newaxis, i, :]) for i in range(input_vec.shape[0])],
                    'samples': [input_vec[i, :] for i in range(input_vec.shape[0])],
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}

        elif type(input_vec) is list:

            if return_dic:
                contents.update({'nn_predictions_on_test_dataset':{'tensors_values': [self._predict_one(vec) for vec in input_vec],
                    'samples': [vec for vec in input_vec],
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}})
                return contents

            return {'tensors_values': [self._predict_one(vec) for vec in input_vec],
                    'samples': [vec for vec in input_vec],
                    'labels_names': [item.name.replace(':', '-') for item in self._pred]}
        else:
            raise ValueError("Wrong type of input argument: {}".format(input_vec))

    def save(self, step: int):
        """
        Save current model and variables into the given output folder.
        It will give an error if the folder already exists.
        """
        self._saver.save(self._sess, self.model_snapshot_save_path, global_step=step)

        print("Saved model to: {}".format(self.model_snapshot_save_path))

    def fit(self, training_samples,
                  training_labels,
                  validation_samples,
                  validation_labels,
                  args, **fit_params):

        prev_acc  = 0

        labels = self.one_hot_encode(training_labels.astype(int), 2)
        save_nn_predictions = PickleTensorValuesSaver(content_name='nn_predictions_on_validation_dataset')
        save_nn_predictions2excel = ExcelTensorValuesSaver(content_name='nn_predictions_on_validation_dataset')
        save_nn_predictions._output_filename = os.path.join(args.output, 'pickles',
            (os.path.basename(args.input_tf_dataset)).split('.')[0] + '-' + "last_prediction" + ".pkl")

        save_nn_predictions2excel._output_filename = os.path.join(args.output, 'excels')

        def plot_grid(step):

            visualizer_grid_and_data_base_on_which_nn_was_trained = Visualizer2D(
                                              output_filename=os.path.join(args.output, 'plots',
                                             (os.path.basename(args.input_tf_dataset)).split('.')[0] + '-' + str(step) + ".html"))

            x = self.transform(validation_samples)
            x = save_nn_predictions.transform({'nn_predictions_on_validation_dataset': x})
            x = save_nn_predictions2excel.transform(x)
            dic = {'training_dataset': {'samples': training_samples, 'labels': training_labels}}
            x.update(dic)
            x = visualizer_grid_and_data_base_on_which_nn_was_trained.transform(x)

        """
        TODO: add ability to load pretrained model
        """

        best_acc = 0

        num_steps = fit_params.get('training_epochs', 1000)
        display_step = fit_params.get('summary_freq', 100)


        # Run the initializer
        self._sess.run(self.init)

        loss, acc, summary = self._sess.run([self.loss_op, self.accuracy, self.merged_summary_op],
                                            feed_dict={self.X: training_samples, self.Y: labels})
        print(acc, "acc", loss, "loss")

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
        if self._pretrained_model:
            print("-- loading pretrained model" * 100)
            self._load_model(self._pretrained_model)
            self._model_loaded = True
            loss, acc, summary = self._sess.run([self.loss_op, self.accuracy, self.merged_summary_op],
                                                feed_dict={self.X: training_samples, self.Y: labels})
            print(acc, "acc", loss, "loss")


            # TODO: uncomment if you want to show classification by pretrained model

        for step in range(1, num_steps + 1):

            _ = self._sess.run(self.train_op, feed_dict={self.X: training_samples, self.Y: labels})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc, summary = self._sess.run([self.loss_op, self.accuracy, self.merged_summary_op],
                                                    feed_dict={self.X: training_samples, self.Y: labels})

                if acc > best_acc:
                    best_acc = acc

                summary_writer.add_summary(summary, step)
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.5f}".format(loss) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc) + ', Learning rate= %f' % (self._sess.run(self.optimizer._lr)) + \
                      ", Best Accuracy= " + "{:.5f}".format(best_acc))


                if prev_acc == best_acc :
                    print("No progress, quit", step)
                    break;

                if best_acc == 1.0:
                    print("Best accuracy achieved", step)
                    self.save(step=step)
                    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name)):
                        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name))

                    with open(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name,'good_results.txt'), 'a') as f:
                        f.write(os.path.dirname(self.logs_path))
                        f.write('\n')
                    print(self.logs_path)
                    break;

                prev_acc = best_acc

            # draw decision boundaries
            #
            if self.verbose:
                if step % (1 * display_step) == 0:
                    self._model_path = os.path.dirname(self.model_snapshot_save_path)
                    plot_grid(step)

        print("Optimization Finished!", step)

        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name)):
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name))

        with open(os.path.join(os.path.dirname(os.path.dirname(self.logs_path)), self.activation_name,'all_results.txt'), 'a') as f:
            f.write(os.path.dirname(self.logs_path) + ' ' + str(acc))
            f.write('\n')
        print(self.logs_path)

        print("Testing Accuracy:", \
              self._sess.run(self.accuracy, feed_dict={self.X: training_samples, self.Y: labels}))

        return best_acc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sess is not None:
            self._sess.close()


    def _load_model(self, save_path, verbose=False):
        """
        Loads a saved TF model from a file.
            Args:
                save_path: The save path of the saved session, returned by Saver.save().
            Returns:
                None
        """
        graph = tf.get_default_graph()
        metagraph_file = None
        for file in sorted(os.listdir(save_path), reverse=True):
            if file.endswith('.meta'):
                metagraph_file = os.path.join(save_path, file)
                break
        if metagraph_file is None:
            raise FileNotFoundError("Cant find metagraph file in: {}".format(save_path))
        tvars = tf.trainable_variables()

        # TODO: fix. We initialize the model for training, but training should only happen when we run fit.
        # there is some weird things going on here.

        self._sess.run(self.init)
        tvars_vals = self._sess.run(tvars)

        if(self.restore_meta_graph):
            print("Loading metagraph from '%s'..." % metagraph_file)
            saver = tf.train.import_meta_graph(metagraph_file, clear_devices=True)
            self._saver = saver

        self._restore_weights(save_path)

    def _restore_weights(self, save_path,  verbose=False):
        """
        Loads a saved TF model weights from a file.
          Args:
            save_path: The save path for the saved model, returned by Saver.save().
          Returns:
            None
        """
        latest_checkpoint_file = tf.train.latest_checkpoint(save_path)
        if latest_checkpoint_file is None:
            raise FileNotFoundError("Cant find checkpoint in path: {}".format(save_path))

        print("Loading model weights from: '%s'..." % latest_checkpoint_file)

        if(not(self.restore_meta_graph)):
            # read tensors one by one to dict
            loaded_tensors_dict= {}
            reader = pywrap_tensorflow.NewCheckpointReader(latest_checkpoint_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                loaded_tensors_dict.update({key+':0': reader.get_tensor(key)})

            # tf.get_collection(tf.GraphKeys.VARIABLES)
            vars = {v.name: v for v in tf.trainable_variables()}
            updated_vars = vars.copy()
            for key, value in vars.items():
                if key in loaded_tensors_dict.keys():
                    try :
                        self._sess.run(tf.assign(ref=vars[key], value=loaded_tensors_dict[key]))
                    except Exception:
                        traceback.print_exc()
                    else:
                        if(self.freeze_pretrained):
                            del(updated_vars[key])
            self.trainable=updated_vars
            self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.step, var_list=self.trainable)
            tvars = tf.trainable_variables()
            tvars_vals = self._sess.run(tvars)

        else:
            self._saver.restore(self._sess, save_path=latest_checkpoint_file)

        self.input_name = 'Placeholder:0'
        self.output_names = ['ArgMax:0', 'Placeholder:0']
        self.output_names.append(self.activation_name + ':0')
        for k in range(1, 11):
            self.output_names.append(self.activation_name + '_' + str(k) +':0')
        self.output_names.append('Identity' + ':0')
        self.output_names.append('Softmax' + ':0')
        self._pred = []
        for output_name in self.output_names:
            self._pred.append(tf.get_default_graph().get_tensor_by_name(output_name))
        self._in = tf.get_default_graph().get_tensor_by_name(self.input_name)


if __name__ == '__main__':
    """
    Run this script to train a model
    """

    parser = argparse.ArgumentParser()

    model_choices = ['base',
                     'test',
                     'arch_type_1',
                     'nn_simple',
                     'nn_9_by_3_simple',
                     'nn_10_by_3_simple',
                     'nn_16_by_3_simple',
                     'nn_18_by_3_simple']

    parser.add_argument('--input-tf-dataset', default='simple.tfrecords', help="tfrecords data set name")
    parser.add_argument('--output', help='Output folder where model is saved')
    parser.add_argument('--model',  default='base',choices=model_choices, help='Specifies which model to use for the image stream.')
    parser.add_argument('--save_freq', default=1, help='frequency of epochs to save model snapshot.')
    parser.add_argument('--validation_freq', default=1, help='frequency of validation of the model in number of iterations.')
    parser.add_argument('--summary_freq', type=int, default=50, help='frequency of validation of the model in number of iterations.')
    parser.add_argument('--pretrained_model', default=None, help='Set pretrained model path if exists. Otherwises start from random')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size in training.')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--training_epochs', type=int, default=1000, help='Number of GPUs to use.')
    parser.add_argument('--use_cpu', action='store_true', help='Use cpu')
    parser.add_argument('--restore_meta_graph', action='store_true', help='')
    parser.add_argument('--freeze_pretrained', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')

    args = parser.parse_args()

    initial_path = args.output


    all_subdirs = [d for d in glob.glob(args.output + '/*') if os.path.isdir(d)]
    if args.pretrained_model and not len(all_subdirs) == 0: # TODO now it is hacked make this as it supposed to be.
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        args.output = os.path.join(initial_path, args.model, timestr + "-pretrained-on-" + os.path.basename(args.pretrained_model) )
        model_path = None
    else:
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        args.output = os.path.join(initial_path, args.model, timestr)
        model_path = None
        print('Created new output directory: ' + args.output)

    fit_params = vars(args)
    nn = NeuralNetworkSimple(model_path=model_path, **fit_params)
    nn.fit(args, **fit_params)