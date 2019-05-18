from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
from transformers.betti_calc import BettiCalc
import csv
import multiprocessing


def run_calculation(csv_file, task, model_path, julia, divisor, neighbors):
    print("Starting task ", task)
    loops = []
    betti_calc = BettiCalc(model_path, julia, divisor, neighbors)
    trace = betti_calc.transform(csv_file)

    loops.append({'trace': trace, 'name': os.path.basename(csv_file)})
    print("Done task ", task)
    return loops


class ExcelToBettiFeeder(Transformer):

    def __init__(self, model_path, julia,  excel_names, divisor, neighbors):
        self.excel_dir = os.path.join(model_path, 'excels')
        self.model_path = model_path
        self.excel_names = excel_names
        self.julia = julia
        self.divisor = int(divisor)
        self.neighbors = int(neighbors)

    def transform(self, content=None):

        betti_calc = BettiCalcParallel(self.model_path, self.julia, self.divisor, self.neighbors)
        loops = []
        for file in self.excel_names:
           csv_file = os.path.join(self.excel_dir, file)
           trace = betti_calc.transform(csv_file)

           loops.append({'trace': trace, 'name' : file})

        content.update({'betti_list' : loops})
        return content
        # return ''