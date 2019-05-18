from typing import List, Union
import numpy as np
from transformers import Transformer
import pickle
import os
import glob

class PickleReader(Transformer):
    def __init__(self, input_filename="saved_dataset.pkl"):
        self._input_filename = input_filename

    def predict(self, content=None):
        """Does not support predict instead reads input
        """

        return self.transform(content)

    def transform(self, content=None):
        if os.path.isdir(self._input_filename):
            print("\nPickle Reader: given  directory: " + self._input_filename)
            file = (sorted(glob.glob(self._input_filename + '/*.pkl')))[-1]

            self._input_filename = os.path.join(self._input_filename, os.path.basename(file))
            print("\nPickle Reader choose file: " + self._input_filename)

        with (open(self._input_filename, "rb")) as openfile:
            a = pickle.load(openfile)


            grid = a['grid']
            predictions = a['predictions']
            y_category = [item[0] for item in predictions]

            # format ([1.0], [-0.3360700011253357, -0.6241478323936462])
            classification_res = [(y_category[i], grid[i]) for i in range(len(y_category))]
        if content is not None:
            dic = {'classification_res' : classification_res, 'dataset': content['dataset']}
        else:
            dic = {'classification_res': classification_res}
        return dic
