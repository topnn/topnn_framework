from argparse import Namespace
from sklearn.pipeline import Pipeline
from transformers.tf_reader import TFReader
from transformers.visualizer_2d import Visualizer2D
import os

def load(args: Namespace) -> Pipeline:
    pipeline_steps = [
        ('reader', TFReader(input_filename=os.path.join(args.input_tf_folder, args.input_tf_dataset))),
        ('visualizer', Visualizer2D(output_filename=os.path.join(args.output_plots_folder, "plot.html"))),
         ]

    return Pipeline(steps=pipeline_steps)