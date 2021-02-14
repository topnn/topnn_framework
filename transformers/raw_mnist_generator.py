import numpy as np
import os
from transformers import Transformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split(data, label, test_size=0.22, sub_sample=70000):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size)

    train_size = int(sub_sample*(1-test_size))
    test_size = int(sub_sample * test_size)

    return ([X_train[0:train_size], y_train[0:train_size] ], [X_test[0:test_size], y_test[0:test_size]])

class MnistParser(Transformer):

    def __init__(self, dim=50, visual=False, load_path="."):
        self.dim = dim
        self.visual = visual
        self.load_path = load_path
        self.classification_digit = 1

    def load_mnist(self):
        data_dir = os.path.join(self.load_path, "mnist")

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        return (X / 255.), y

    def transform(self, content=None):

        X, y = self.load_mnist()
        flattened_images = X.reshape(X.shape[0], -1)
        ytwo_classes = []

        for i, label in enumerate(y):

           if label == self.classification_digit:
               ytwo_classes.append(1)
           else:
               ytwo_classes.append(0)

        pca = PCA(n_components=self.dim)
        pca.fit(flattened_images)
        X = pca.transform(flattened_images)
        X_reconstructed = pca.inverse_transform(X)

        return [X, ytwo_classes, X_reconstructed]


if __name__ == "__main__":

    mp = MnistParser(load_path='/home/greg/topology/db/')
    X, y = mp.load_mnist()
    X_pca, y_pca, X_reconstructed = mp.transform()
    trn, tsts = split(X_reconstructed, y_pca, test_size=1.0/7.0)
    first_image = trn[0][0]
    pixels = first_image.reshape((28, 28))

    plt.imshow(pixels, cmap='gray')
    plt.title(trn[1][0])
    plt.show()