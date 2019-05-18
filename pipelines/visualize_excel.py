from argparse import Namespace
from sklearn.pipeline import Pipeline
from transformers.excel_to_plot_feeder import ExcelToPlotFeeder

def load(args: Namespace) -> Pipeline:
    log_path = args.output
    fit_params = vars(args)

    pipeline_steps = [ ('excel_to_betti_feeder', ExcelToPlotFeeder(model_path=args.pretrained, excel_names=[
                                                                # 'cat1_LeakyRelu-0.csv',
                                                                # 'cat1_LeakyRelu_1-0.csv',
                                                                # 'cat1_LeakyRelu_2-0.csv',
                                                                'cat1_LeakyRelu_3-0.csv',
                                                                # 'cat1_LeakyRelu_4-0.csv',
                                                                # 'cat1_LeakyRelu_5-0.csv',
                                                                # 'cat1_LeakyRelu_6-0.csv',
                                                                # 'cat1_Relu-0.csv',
                                                                # 'cat1_Relu_1-0.csv',
                                                                # 'cat1_Relu_2-0.csv',
                                                                # 'cat2_Relu_2-0.csv',
                                                                # 'cat1_Relu_3-0.csv',
                                                                # 'cat1_Relu_4-0.csv',
                                                                # 'cat1_Relu_5-0.csv',
                                                                # 'cat1_Relu_6-0.csv',
                                                                # 'cat1_LeakyRelu_7-0.csv',
                                                                # 'cat1_LeakyRelu_8-0.csv',
                                                                # 'cat1_LeakyRelu_9-0.csv'
                                                                ]))
    ]

    return Pipeline(steps=pipeline_steps)