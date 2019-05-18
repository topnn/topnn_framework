from typing import List, Union
import numpy as np
from transformers import Transformer
import csv
import os

class ExcelTensorValuesSaver(Transformer):
    def __init__(self, content_name, output_filename="saved_dataset.csv", ):
        self._output_filename = output_filename
        self.content_name = content_name

        if not os.path.exists(output_filename):
            os.makedirs(output_filename)

    def transform(self, content=None):

        if type(content) == type({}):
            # create directory if does not exist
            if not os.path.exists(os.path.join(os.path.dirname(self._output_filename))):
                os.makedirs(os.path.dirname(self._output_filename))

            tensor_names = content[self.content_name]['labels_names']
            tensor_names = [ name.replace('/','') for name in tensor_names]
            predictions = content[self.content_name]['labels']
            samples = content[self.content_name]['samples']
            samples_n = len(samples)

            # first tensor is ArgMax:0 it is used to infer category
            for j in range(1, len(tensor_names)):

                name = tensor_names[j]
                tensor_dim = predictions[0][j].shape[1]
                tensor_values_all = np.zeros(shape=(samples_n, tensor_dim))
                tensor_values_cat1 = np.zeros(shape=(samples_n, tensor_dim))
                tensor_values_cat2 = np.zeros(shape=(samples_n, tensor_dim))

                for i, point in enumerate(samples):

                    tensor_values_all[i, :] = predictions[i][j]

                    if predictions[i][0] == 1:
                        tensor_values_cat1[i, :] = predictions[i][j]

                    elif predictions[i][0] == 0:
                        tensor_values_cat2[i, :] = predictions[i][j]

                file_name = 'cat1' + '_' + name + '.csv'
                with open(os.path.join(self._output_filename, file_name), 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                    for row in np.transpose(tensor_values_cat1):
                        spamwriter.writerow(row)

                file_name = 'cat2' + '_' + name + '.csv'
                with open(os.path.join(self._output_filename, file_name), 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for row in np.transpose(tensor_values_cat2):
                        spamwriter.writerow(row)

                file_name = 'all' + '_' + name + '.csv'
                with open(os.path.join(self._output_filename, file_name), 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for row in np.transpose(tensor_values_all):
                        spamwriter.writerow(row)

            return content
        else:
            return ''
    def predict(self, content=None):
        """saver done't support predict instead writes content on the disk"""

        return self.transform(content)
