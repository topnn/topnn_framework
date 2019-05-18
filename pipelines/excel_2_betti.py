from argparse import Namespace
from sklearn.pipeline import Pipeline
import os
from transformers.visualizer_2d_loops import Visualizer2DLoops
from transformers.excel_to_betti_feeder_parallel import ExcelToBettiFeederParallel
from transformers.constant_feeder import ConstantFeeder
from transformers.grid_generator import GridGenerator
from estimators.nn_simple import NeuralNetworkSimple
from transformers.excel_tensor_values_saver import ExcelTensorValuesSaver

def load(args: Namespace) -> Pipeline:
    log_path = args.output
    fit_params = vars(args)
    activation_name = args.activation_type

    if args.cat2:
        excels_cat_2_a = ['cat2_Placeholder-0.csv', 'cat2_%s-0.csv' % (activation_name)]
        excels_cat_2_b = ['cat2_%s_%s-0.csv' % (activation_name, i) for i in range(1, int((args.model.split('_')[0])) - 1)]
        excels_cat_2_c = ['cat2_Identity-0.csv']

        excels_cat_2 = excels_cat_2_a + excels_cat_2_b + excels_cat_2_c
    else:
        excels_cat_2 = []

    if args.cat1:
        excels_cat_1_a = ['cat1_Placeholder-0.csv', 'cat1_%s-0.csv' % (activation_name)]
        excels_cat_1_b = ['cat1_%s_%s-0.csv' % (activation_name, i) for i in
                          range(1, int((args.model.split('_')[0])) - 1)]
        excels_cat_1_c = ['cat1_Identity-0.csv']

        excels_cat_1 = excels_cat_1_a + excels_cat_1_b + excels_cat_1_c
        
    else:
        excels_cat_1 = []

    pipeline_steps = [  ('tensor_names', ConstantFeeder(constant={})),
                        ('excel_to_betti_feeder', ExcelToBettiFeederParallel(model_path=log_path, julia=fit_params['julia'],
                        excel_names=excels_cat_2 + excels_cat_1, divisor = fit_params['divisor'], neighbors= fit_params['neighbors'], max_dim=fit_params['betti_max']))
    ]

    return Pipeline(steps=pipeline_steps)
