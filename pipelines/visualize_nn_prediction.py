from argparse import Namespace
from sklearn.pipeline import Pipeline
from transformers.grid_generator import GridGenerator
from transformers.tf_saver import  TFSaver
from estimators.nn import NeuralNetwork
import os
from transformers.pickle_tensors_values_saver import PickleTensorValuesSaver
from transformers.visualizer_2d import Visualizer2D
from transformers.pickle_reader import PickleReader
from transformers.tf_reader import TFReader

def load(args: Namespace) -> Pipeline:
    pipeline_steps = [
        ('generate_grid', GridGenerator(res=0.05)),
        ('nn', NeuralNetwork(is_training=False, architecture=args.model, model_path=args.pretrained_model)),
        ('save_nn_predictions', PickleTensorValuesSaver(output_filename=os.path.join(args.output_pickle_folder, args.output_pickle_predictions))),
        ('read_data_base_on_which_nn_was_trained', TFReader(input_filename=os.path.join(args.input_tf_folder, args.input_tf_dataset))),
        ('read_nn_predictions', PickleReader(input_filename=os.path.join(args.output_pickle_folder, args.output_pickle_predictions))),
        ('visualizer_grid_and_data_base_on_which_nn_was_trained', Visualizer2D(output_filename=os.path.join(args.output_plots_folder, args.input_tf_dataset + ".html")))
         ]

    return Pipeline(steps=pipeline_steps)