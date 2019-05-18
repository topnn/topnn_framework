from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
from transformers.tensors_values_plotter import TensorsValuesPlotter
import numpy as np
import csv
from itertools import combinations


class ExcelToPlotFeeder(Transformer):

    def __init__(self, model_path, excel_names):
        self.excel_dir = os.path.join(model_path, 'excels')
        self.model_path = model_path
        self.excel_names = excel_names

        self.plotter = TensorsValuesPlotter(content_name='results_plot',
                             output_filename=os.path.join(model_path, 'plots',
                                                          'tensors_values'))
    def transform(self, content=None):

        def read_csv(csv_file):
            with open(csv_file) as file:
                readCSV = csv.reader(file, delimiter=',')
                data = []
                for row in readCSV:
                    data.append([float(item) for item in row])

            data = np.transpose(np.array(data))
            return data

        for file in self.excel_names:
           csv_file = os.path.join(self.excel_dir, file)
           data = read_csv(csv_file)

           labels = data
           # have to mach data structure of "TensorsValuesPlotter"
           labels = [ [[item]] for item in data]
           contents = {'results_plot': {'labels_names' :[file],
                            'labels' : labels }}

           for a in list(combinations(range(6), 3 )):
               contents = {'results_plot': {'labels_names': [file  + '_' + str(a[0]) + str(a[1]) + str(a[2])],
                                            'labels': labels}}

               self.plotter.transform(contents, dim_1=a[0], dim_2=a[1], dim_3=a[2], auto_open=True)
