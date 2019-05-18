from __future__ import print_function
from abc import ABC
import os
import time
import glob
import argparse
import numpy as np
from transformers.grid_generator import GridGenerator
import shutil

class NeuralNetworkTrainer(ABC):
    def __init__(self,
                 nn,
                 content_name,
                 args,
                 **fit_params):

        self.args = args
        self.fit_params = fit_params
        self.nn = nn
        self.content_name = content_name

    def transform(self, content):

        def _get_lables_and_data_tf(ds):
            """ parse tf records to points and labels. """

            labels = np.array(ds['labels'])
            labels.shape = (labels.shape[0],)
            samples = np.array(ds['samples'])
            return samples, labels

        training_samples, training_labels = _get_lables_and_data_tf(content['training_dataset'])
        validation_samples, validation_labels = _get_lables_and_data_tf(content['validation_dataset'])

        best_acc = self.nn.fit(training_samples=training_samples,
                    training_labels=training_labels,
                    validation_samples=validation_samples,
                    validation_labels=validation_labels,
                    args=self.args,
                    **self.fit_params)

        if best_acc == 1.0:
        # if True:
            dic = {self.content_name : self.nn.transform(content['test_dataset']['samples'])}
            content.update(dic)
            return content
        else:
            # remove simulation directory if not successful
            print("accuracy less than 100 percent, remove directory:", self.fit_params['output'])
            if os.path.basename(self.fit_params['output'])[0:4] == '2019': # protection to insure we delete diretories that start with 2019
                shutil.rmtree(self.fit_params['output'])
            else:
                print('WARNING! - attempted to delete and discovered  output directory is weired')
            return ''


    def fit(self, content):
        # no fit just does transform
        return self.transform(content)

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
                     'nn_12_by_3_simple',
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

    else:
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        args.output = os.path.join(initial_path, args.model, timestr)
        print('Created new output directory: ' + args.output)

    fit_params = vars(args)
    nn_trainer = NeuralNetworkTrainer(args, **fit_params)
    nn_trainer.transform([])
