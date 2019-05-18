from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
import julia

class ConstantFeeder(Transformer):
    def __init__(self, constant):
        self.constant = constant
        
    def transform(self, content=None):
        return self.constant