from argparse import Namespace
from sklearn.pipeline import Pipeline
from transformers.tf_reader import TFReader
from transformers.visualizer_2d import Visualizer2D
from transformers.constant_feeder import ConstantFeeder

import os

def load(args: Namespace) -> Pipeline:
    pipeline_steps = [
        ('contents', ConstantFeeder({})),
        ('training_dataset_circles', TFReader(input_filename=args.input_tf_dataset,
                                              content_name='training_dataset')),
        ('visualizer', Visualizer2D(output_filename=os.path.join(args.output_plots_folder, "plot.html"))),
         ]

    return Pipeline(steps=pipeline_steps)
