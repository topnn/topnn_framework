from argparse import Namespace
from sklearn.pipeline import Pipeline
from estimators.nn_trainer import NeuralNetworkTrainer
import os
from transformers.pickle_tensors_values_saver import PickleTensorValuesSaver
from transformers.visualizer_2d import Visualizer2D
from transformers.visualizer_2d_loops import Visualizer2DLoops
from transformers.tf_reader import TFReader
from transformers.excel_tensor_values_saver import ExcelTensorValuesSaver
from transformers.tensors_values_plotter import TensorsValuesPlotter
from transformers.constant_feeder import ConstantFeeder
from transformers.excel_to_betti_feeder import ExcelToBettiFeeder
from estimators.nn_simple import NeuralNetworkSimple
from transformers.grid_generator import GridGenerator

def load(args: Namespace) -> Pipeline:
    fit_params = vars(args)

    # directory to save training results and various calculations
    log_path = args.output

    # neural network to be trained and evaluated
    nn = NeuralNetworkSimple(model_path=os.path.join(log_path, 'snapshots'), num_input = 2, **fit_params)

    pipeline_steps = [

        # dummy dictionary to which various transformers will append their contents
        ('contents', ConstantFeeder({})),

        # generate training dataset used to train neural network
        ('training_dataset_circles', TFReader(input_filename=fit_params.get('input_tf_dataset'),
                                              content_name='training_dataset')),

        # when verbose nn will run on validation dataset to see how decision boundaries change
        ('validation_dataset_grid', GridGenerator(res=0.05, content_name='validation_dataset')),

        # once nn is finished training it will run on test set, this is the
        # set used for betti numbers calculation
        ('test_dataset_grid', GridGenerator(res=0.025, content_name='test_dataset')),
        # ('test_dataset_grid', TFReader(input_filename= os.path.join(os.path.dirname(fit_params.get('input_tf_dataset')), 'inference_' + os.path.basename(fit_params.get('input_tf_dataset'))),
        #                                content_name='test_dataset', mode='test')),
        # wrapper to:
        # -train nn on content['training_dataset']
        # -when verbose display results on content['validation_dataset']
        # -when finished fitting run on test on content['test_dataset']
        ('nn_trainer', NeuralNetworkTrainer(nn=nn,
                                            content_name='nn_predictions_on_test_dataset',
                                            args=args,
                                            **fit_params)),

        # # dump tensor values to pickle file
        # ('save_content_to_pickle', PickleTensorValuesSaver(output_filename=os.path.join(log_path,
        #                                                    'pickles',
        #                                                    'nn_after_fit.pkl'),
        #                                                     content_name='nn_predictions_on_test_dataset')),
        # dump tensor values to excel file
        ('save_content_to_excel', ExcelTensorValuesSaver(output_filename=os.path.join(log_path, 'excels'),
                                                          content_name='nn_predictions_on_test_dataset')),
        #
        # plot tensor values
        ('layer_output_plotter', TensorsValuesPlotter(content_name='nn_predictions_on_test_dataset',
                                                      output_filename=os.path.join(log_path, 'plots', 'tensors_values'),
                                                      enable=args.plot
                                                      )),

        # # visualize training and test set
        ('training_and_test_dataset_visualizer',  Visualizer2D(output_filename=os.path.join(log_path, 'plots',
                      os.path.basename(args.input_tf_dataset).split('.')[0] + '-trainer' + ".html"),
                                                               enable=args.plot
                                                               ))
    ]

    return Pipeline(steps=pipeline_steps)