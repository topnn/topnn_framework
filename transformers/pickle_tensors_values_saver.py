from typing import List, Union
import numpy as np
from transformers import Transformer
import pickle
import os

class PickleTensorValuesSaver(Transformer):
    def __init__(self, content_name, output_filename="saved_dataset.pkl"):
        self._output_filename = output_filename
        self.content_name = content_name

    def transform(self, content=None):

        # create directory if does not exist
        if not os.path.exists(os.path.dirname(self._output_filename)):
            os.makedirs(os.path.dirname(self._output_filename))

        with open(self._output_filename, 'wb') as f:
            pickle.dump(content[self.content_name], f)

        return content

    def predict(self, content=None):
        """saver done't support predict instead writes content on the disk"""

        return self.transform(content)
