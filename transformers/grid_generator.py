from typing import List, Union
import numpy as np
from transformers import Transformer
import matplotlib.pyplot as plt

class GridGenerator(Transformer):
    def __init__(self, content_name ='grid', random=False, grid_min=-1.1, grid_max=1.1, res=0.1,  mode='2D'):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.res = res
        self.content_name = content_name
        self.random = random
        self.mode =  mode

    @staticmethod
    def gen_grid(grid_min, grid_max, res, random=False, mode='2D'):

        if mode == '2D':
            x = np.arange(grid_min, grid_max, res)

            y = np.arange(grid_min, grid_max, res)

            xx, yy = np.meshgrid(x, y)
            if random:
                xx = xx + np.random.randn(xx.shape[0], xx.shape[1]) * 0.02
                yy = yy + np.random.randn(yy.shape[0], yy.shape[1]) * 0.02
            grid = np.dstack((xx, yy)).reshape(-1, 2)

        if mode == '3D':
            x = np.arange(grid_min, grid_max, res)
            y = np.arange(grid_min, grid_max, res)
            z = np.arange(grid_min, grid_max, res)

            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            grid = np.stack((xx, yy, zz), axis=3).reshape(-1, 3)
            print("-" * 100, "length of  grid", len(grid))

        return grid

    def transform(self, content=None):
        grid = self.gen_grid(self.grid_min, self.grid_max, self.res, self.random, self.mode)
        dic = {self.content_name : {'samples': grid,
               'labels': [0] * grid.shape[0]} }

        if content != None:
            content.update(dic)
        else:
            content = dic

        return content


    
      
