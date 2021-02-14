from typing import List, Union
import numpy as np
from transformers import Transformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split(data, label, test_size=0.22, sub_sample=10000):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size)

    train_size = int(sub_sample*(1-test_size))
    test_size = int(sub_sample * test_size)

    return ([X_train[0:train_size], y_train[0:train_size] ], [X_test[0:test_size], y_test[0:test_size]])

class CirclesGenerator(Transformer):
    def __init__(self, content_name ='circles', random=False, grid_min=-10, grid_max=10, res=0.19,  mode='2D'):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.res = res
        self.content_name = content_name
        self.random = random
        self.mode =  mode

    def gen_grid(self, grid_min, grid_max, res, random=False, centers=[(0,0), (8,8)], radius=0.5, margin=0.5):

        x = np.arange(grid_min, grid_max, res)
        y = np.arange(grid_min, grid_max, res)
        xx, yy = np.meshgrid(x, y)
        if random:
            xx = xx + np.random.randn(xx.shape[0], xx.shape[1]) * 0.02
            yy = yy + np.random.randn(yy.shape[0], yy.shape[1]) * 0.02
        grid = np.dstack((xx, yy)).reshape(-1, 2)

        y = np.ones(shape=(len(grid), 2)) * 1

        for center in centers:
            for i, point in enumerate(grid):

                if np.linalg.norm(center - point) < radius + margin:
                    y[i, 0] = 3
                    y[i, 1] = 3

                if np.linalg.norm(center - point) < radius:
                    y[i, 0] = 0 * y[i, 0]
                    y[i, 1] = not y[i, 0]

            y = np.array(y)

            mask = (y == 3)
            label = y[:, 0]
            label = np.delete(label, np.where(mask[:, 0]))
            xx = list(grid[:, 0])
            yy = list(grid[:, 1])

            xx = np.delete(xx, np.where(mask[:, 0]))
            yy = np.delete(yy, np.where(mask[:, 0]))
            grid = np.ones(shape=(len(xx), 2))

            grid[:, 0] = xx
            grid[:, 1] = yy

            y = np.ones(shape=(len(xx), 2))
            y[:,0] = label
            y[:, 1]= 1- label

        return ( grid,  y)

    def transform(self, content=None, big_r = 7, small_r = 1.3, n=9):

        import math
        pi = math.pi

        def LargeCirlce(r, n=n):
            return [(math.cos(2 * pi / n * x) * r, math.sin(2 * pi / n * x) * r) for x in range(0, n + 1)]

        X, y = self.gen_grid(self.grid_min, self.grid_max, self.res, self.random, centers=list(LargeCirlce(big_r, n-1)) + [0,0], radius=small_r)


        return [X, y]


    
if __name__ == "__main__":

    mp = CirclesGenerator()
    content = mp.transform()
    X = content[0]
    y = content[1]
    trn, tsts = split(X, y)

    plt.scatter(trn[0][:, 0], trn[0][:, 1], c=trn[1][:, 0])
    plt.show()