import numpy as np
from transformers import Transformer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import math
import random
from sklearn import preprocessing
def split(data, label, test_size=0.22, sub_sample=60000):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size)

    train_size = int(sub_sample*(1-test_size))
    test_size = int(sub_sample * test_size)

    return ([X_train[0:train_size], y_train[0:train_size] ], [X_test[0:test_size], y_test[0:test_size]])



class DataSet3DSpheresGenerator(Transformer):

    def __init__(self, visual=False, samples=9450):
        self.visual = visual
        self.samples = int(samples/9)

    def fibonacci_sphere(self, samples=2000, randomize=True, radius=1.0):
        rnd = 1.
        if randomize:
            rnd = random.random() * samples

        points = []
        offset = 2. / samples
        increment = math.pi * (3. - math.sqrt(5.))

        for i in range(samples):

            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % samples) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x * radius, y * radius, z * radius])

        return points

    def gen_2spheres(self, N_SAMPLES=1000, visual=True, r1=1, r2=1.05, r3=2, r4=2.05, r5=0.5):

        random.seed(1)
        x1a = self.fibonacci_sphere(samples=N_SAMPLES, randomize=True, radius=r1)
        y1a = np.zeros(len(x1a)).tolist()

        x1b = self.fibonacci_sphere(samples=N_SAMPLES, randomize=False, radius=r2)
        y1b = np.zeros(len(x1b)).tolist()

        x2a = self.fibonacci_sphere(samples=N_SAMPLES, randomize=False, radius=r3)
        y2a = np.ones(len(x2a)).tolist()

        x2b = self.fibonacci_sphere(samples=N_SAMPLES, randomize=False, radius=r4)
        y2b = np.ones(len(x2b)).tolist()

        x3 = self.fibonacci_sphere(samples=int(N_SAMPLES), randomize=False, radius=r5)
        y3 = np.ones(len(x3)).tolist()

        x = np.asarray(x1a + x1b + x2a + x2b + x3)
        y = y1a + y1b + y2a + y2b + y3

        return x, y

    def transform(self, content=None):

        def gen_spheres(N_SAMPLES, visual, center, flip):

            X, y = self.gen_2spheres(N_SAMPLES=N_SAMPLES, visual=visual)
            X = X + center
            y = np.abs(flip - np.array(y))

            return X, y

        X1, y1 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(0,0,0), flip =0)
        X2, y2 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(-3,-3,-3), flip =0)
        X3, y3 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(3, 3, 3), flip =0)

        X4, y4 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(-3, 3, 3), flip =0)
        X5, y5 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(-3, 3, -3), flip =0)
        X6, y6 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(-3,-3, 3), flip =0)

        X7, y7 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(3, -3, -3), flip =0)
        X8, y8 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(3, -3, 3), flip =0)
        X9, y9 = gen_spheres(N_SAMPLES=self.samples, visual=False, center=(3,  3, -3), flip =0)

        X_total = np.concatenate((X1, X2), axis=0)
        y_total = np.concatenate((y1, y2), axis=0)

        X_total = np.concatenate((X_total, X3), axis=0)
        y_total = np.concatenate((y_total, y3), axis=0)

        X_total = np.concatenate((X_total, X4), axis=0)
        y_total = np.concatenate((y_total, y4), axis=0)

        X_total = np.concatenate((X_total, X5), axis=0)
        y_total = np.concatenate((y_total, y5), axis=0)

        X_total = np.concatenate((X_total, X6), axis=0)
        y_total = np.concatenate((y_total, y6), axis=0)

        X_total = np.concatenate((X_total, X7), axis=0)
        y_total = np.concatenate((y_total, y7), axis=0)

        X_total = np.concatenate((X_total, X8), axis=0)
        y_total = np.concatenate((y_total, y8), axis=0)

        X_total = np.concatenate((X_total, X9), axis=0)
        y_total = np.concatenate((y_total, y9), axis=0)

        X = X_total.copy()
        y = y_total.copy()

        max_abs_scaler = preprocessing.MaxAbsScaler()
        X = max_abs_scaler.fit_transform(X)

        return [X, y]


if __name__ == '__main__':
    spheres_generator = DataSet3DSpheresGenerator(visual=True)
    content = spheres_generator.transform()
    X = content[0]
    y = content[1]
    trn, tsts = split(X, y, test_size=1-4.0/5.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trn[0][:, 0], trn[0][:, 1], trn[0][:, 2], c=trn[1])
    plt.show()
    print(len(trn[1]))

