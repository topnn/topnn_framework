import numpy as np
from transformers import Transformer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

def split(data, label, test_size=0.22, sub_sample=60000):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size)

    train_size = int(sub_sample*(1-test_size))
    test_size = int(sub_sample * test_size)

    return ([X_train[0:train_size], y_train[0:train_size] ], [X_test[0:test_size], y_test[0:test_size]])



class DataSet3DRingsGenerator(Transformer):
    def __init__(self, visual=False, samples=15000, shapeParam1=15, shapeParam2=2.5, shapeParam3=2.2, radius=1):
        self.visual = visual
        self.shapeParam1 = shapeParam1
        self.shapeParam2 = shapeParam2
        self.shapeParam3 = shapeParam3
        self.radius = radius
        self.samples= samples
        self.range = 0.5

    def draw_circle(self, r, center, n, rand=True):

        angles = np.linspace(start=0, stop=n, num=n) * (np.pi * 2) / n
        X = np.zeros(shape=(n, 2))
        X[:, 0] = np.sin(angles) * r
        X[:, 1] = np.cos(angles) * r

        if rand:
            return X + center + np.random.rand(n, 2) * r / self.shapeParam1
        else:
            return X + center

    def gen_ring(self, center, flip, q=1.4, r=1):

        N_SAMPLES = self.samples
        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)
        X2 = self.draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)


        X[0:N_SAMPLES, 0] = (X1[:, 0]) * self.shapeParam2 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[0:N_SAMPLES, 1] = (X1[:, 1]) * self.shapeParam2 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[0:N_SAMPLES, 2] = np.random.uniform(low=-self.range, high=self.range, size = X1.shape[0]) * q

        X[N_SAMPLES: 2 * N_SAMPLES, 0] = X2[:, 0] * self.shapeParam3 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[N_SAMPLES: 2 * N_SAMPLES, 1] = X2[:, 1] * self.shapeParam3 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[N_SAMPLES: 2 * N_SAMPLES, 2] = np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q

        y[:] = flip
        y[0:N_SAMPLES] = flip

        X_total = X.copy() + np.array(( self.shapeParam3, 0, 0))
        y_total = y.copy()

        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)
        X2 = self.draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)

        X[0:N_SAMPLES, 0] = (X1[:, 0]) * self.shapeParam2 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[0:N_SAMPLES, 2] = (X1[:, 1]) * self.shapeParam2 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[0:N_SAMPLES, 1] = np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q

        X[N_SAMPLES: 2 * N_SAMPLES, 0] = X2[:, 0] * self.shapeParam3 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[N_SAMPLES: 2 * N_SAMPLES, 2] = X2[:, 1] * self.shapeParam3 + np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q
        X[N_SAMPLES: 2 * N_SAMPLES, 1] = np.random. uniform(low=-self.range, high=self.range,size = X1.shape[0]) * q

        y[:] = 1 - flip
        y[0:N_SAMPLES] = 1 - flip

        X_total = np.concatenate((X_total, X), axis=0) + center
        y_total = np.concatenate((y_total, y), axis=0)

        return X_total, y_total

    def transform(self, q=3):

        X1, y1 = self.gen_ring((q, q, q), 0)
        X2, y2 = self.gen_ring((-q, -q, q), 1)
        X3, y3 = self.gen_ring((-q, q, -q), 0)
        X4, y4 = self.gen_ring((q, -q, -q), 1)
        X5, y5 = self.gen_ring((0, 0, 0), 0)
        X6, y6 = self.gen_ring((-q, -q, -q), 0)
        X7, y7 = self.gen_ring((q, q, -q), 1)
        X8, y8 = self.gen_ring((-q, q, q), 0)
        X9, y9 = self.gen_ring((q, -q, q), 1)

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

    mp = DataSet3DRingsGenerator()
    content = mp.transform()
    X = content[0]
    y = content[1]
    trn, tsts = split(X, y, test_size=1-4.5/6.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trn[0][:,0], trn[0][:,1], trn[0][:,2], c=trn[1])
    plt.show()


