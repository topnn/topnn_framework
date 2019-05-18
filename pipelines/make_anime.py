from argparse import Namespace
from sklearn.pipeline import Pipeline
from estimators.nn_trainer import NeuralNetworkTrainer
import os
from transformers.pickle_tensors_values_saver import PickleTensorValuesSaver
from transformers.visualizer_2d import Visualizer2D
from transformers.pickle_reader import PickleReader
from transformers.tf_reader import TFReader
from transformers.excel_reader import ExcelReader
from transformers.layer_output_plotter import LayerOutputPlotter
# from transformers.betti_calc import BettiCalc

def load(args: Namespace) -> Pipeline:
    fit_params = vars(args)
    pipeline_steps = [
        # read predictions ad for each tensor store prediction results in excel:
        ('excel_reader', ExcelReader(input_filename=args.output))

        # visualization part:
        #('read_data_base_on_which_nn_was_trained', TFReader(input_filename=args.input_tf_dataset)),
        #('read_nn_predictions', PickleReader(input_filename=os.path.join(args.output, 'pickles'))),
        #('visualizer_grid_and_data_base_on_which_nn_was_trained',
        #Visualizer2D(output_filename=os.path.join(args.output,'plots',
        #              os.path.basename(args.input_tf_dataset).split('.')[0] + '-trainer' + ".html")))
        #('betti_calculator', BettiCalc(args.output))
        ]

    return Pipeline(steps=pipeline_steps)