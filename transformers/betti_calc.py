from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
import julia
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv
import pickle
import time
from sklearn.manifold import Isomap
import sklearn.utils.graph_shortest_path as gp
import networkx as nx

def print_time(str, now):
    print(str, time.time() - now)
    now = time.time()
    return now

class BettiCalc(Transformer):
    def __init__(self, model_path, julia, divisor, neighbors):
        self.betti_dir = os.path.join(model_path, 'betti')
        self.pickle_file =  os.path.join(model_path, 'betti', "workspace_betti.pkl")
        if not os.path.exists(self.betti_dir):
            os.makedirs(self.betti_dir)

        self.fn = julia.include('./julia_include/julia_aux.jl')
        self.fn2 = julia.include('./julia_include/julia_aux2.jl')

        self.verbose = True
        self.number_of_neighbors = int(neighbors)
        self.divisor = int(divisor)

    def transform(self, content=None):

        csv_file = content
        betti_file = os.path.join(self.betti_dir, 'betti-' + os.path.basename(csv_file))

        def read_csv(csv_file, divisor):
            with open(csv_file) as file:
                readCSV = csv.reader(file, delimiter=',')
                data = []
                for row in readCSV:
                    data.append([float(item) for item in row[::divisor]])

            data = np.unique(np.transpose(np.array(data)), axis=0)
            return data

        def find_neighbors(X, n_neighbors):
            """ find n_neighbors nearest neighbors to each point in point cloud X."""
          
            now = time.time()
            now = print_time('NearestNeighbors', now)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                    algorithm='ball_tree').fit(X)
            now = print_time('NearestNeighbors - done', now)
            distances, indices = nbrs.kneighbors(X)
            now = print_time('kneighbors - done',now)
            A = nbrs.kneighbors_graph(X).toarray()
            now = print_time('kneighbors_graph - done', now)
            return indices, A

        def plot_data(data, fig):
            """plot first 3d of data"""

            if data.shape[1] > 3:
                print("Warning: data dimension is larger than 3, dim is %s" % (data.shape[1]))

            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', s=0.5)
            return ax

        def plot_neighbors(data, indices, ax):
            """connect neighboring nodes in data by line """

            indices = [item.tolist() for item in indices]
            counter_removed = 0

            for j in range(data.shape[0]):
                def remove_repetitions(indices, j, counter_removed):
                    for item in indices[j]:
                        try:
                            indices[item].remove(j)
                            counter_removed = counter_removed + 1
                        except ValueError:
                            pass
                    # print("removed", counter_removed)
                    return counter_removed, indices

                counter_removed, indices = remove_repetitions(indices, j, counter_removed)
                origin = data[j, :]
                targets = [data[i, :] for i in indices[j]]

                for target in targets:
                    x = [origin[0], target[0]]
                    y = [origin[1], target[1]]
                    z = [origin[2], target[2]]
                    ax.plot(x, y, z, 'ro-', linewidth='1', markersize=1)


        def creat_graph_and_calc_dist_verb(A):
            """
            creates graph from ajacency matrix and calculates shortest path
            """
            return gp.graph_shortest_path(A, method='auto', directed=False)


        def distance_mat_verb(q, A, cut_off = 10):
            """ Use graph A to build graph distance matrix"""

            # maximum distance between node is 1000


            mat = np.ones(shape=A.shape) * 1000
            for node in range(len(q)):
                for neighbor in range(len(q)):

                    # distance of node to itself is zero
                    if node == neighbor:
                        mat[node, neighbor] = 0
                        continue

                    # nodes that cannot be reached (cut_off distance away) are set to zero which
                    # this problem is fixed below
                    if q[node, neighbor] > cut_off or q[node, neighbor]  == 0:
                        mat[node, neighbor] = 1000
                    else:
                        mat[node, neighbor] = q[node, neighbor]
            return mat

        betti_file = os.path.join(self.betti_dir, 'betti-' + os.path.basename(csv_file))
        betti_curve_file = os.path.join(self.betti_dir, 'betti-curve-' + os.path.basename(csv_file))


        print("running calculation on neighbors filtration")
        data = read_csv(csv_file, self.divisor)

        now = time.time()
        now = print_time('Start: creat neighborhood graph and calculate graph distance (ISOMAP)', now)
        indices, A = find_neighbors(data, self.number_of_neighbors)
        mat = creat_graph_and_calc_dist_verb(A)
        mat = distance_mat_verb(mat, A)
        now = print_time('Done: creat neighborhood graph and calculate graph distance (ISOMAP)', now)

        if self.verbose:
            pass        

        neihbors_betti_file = os.path.join(self.betti_dir, 'betti-nbr-' + str(self.divisor) + '-' +  str(self.number_of_neighbors) + '-' + os.path.basename(csv_file))
        neihbors_curve_file = os.path.join(self.betti_dir, 'betti-nbr-curve-' + str(self.divisor) + '-' +  str(self.number_of_neighbors) + '-' + os.path.basename(csv_file))
        file_repres_file = os.path.join(self.betti_dir, 'betti-nbr-repres-' + str(self.divisor) + '-' +  str(self.number_of_neighbors) + '-' + os.path.basename(csv_file).split('.')[0] + '.txt')
        mat_file = os.path.join(self.betti_dir, 'mat' + str(self.divisor) + '-' +  str(self.number_of_neighbors) + '-' + os.path.basename(csv_file).split('.')[0])

        maxdim = 2
        np.save(mat_file, mat)
        print("fn2 call :", "neihbors_betti_file", neihbors_betti_file, "neihbors_curve_file", neihbors_curve_file, "file_repres_file",  file_repres_file,
              "maxdim", maxdim, "mat_file", mat_file)

        print('"'+ neihbors_betti_file + '",' + '"' +neihbors_curve_file+ '",' +  '"' +file_repres_file + '",'+ str(maxdim) + ','  '"'+ mat_file + '.npy"' )

        self.fn2(neihbors_betti_file, neihbors_curve_file, file_repres_file, maxdim, mat_file + ".npy")

        return 'done' # uncomment to skip loops draw

        file_repres_file = file_repres_file + '_1'
        with open(file_repres_file) as f:
            lines = f.readlines()

        def draw_repres(data, origin, target, ax, k):
            """ Add connection from origin to targets"""

            colors = ['y', 'g', 'k', 'm', 'c', 'b']
            for t in range(len(origin)):
                x = [data[origin[t] - 1, 0], data[target[t] - 1, 0]]
                y = [data[origin[t] - 1, 1], data[target[t] - 1, 1]]
                z = [data[origin[t] - 1, 2], data[target[t] - 1, 2]]
                ax.plot(x, y, z, colors[k % 5] + 'o-', linewidth='3', markersize=1)

        def get_origin_and_targets(line, divisor):
            t = line.split(" ")
            q = [item.replace("'", "").replace("[", "").replace("]", "").replace(";", "").replace("\n", "") for item in
                 t]
            points = [(int(item) - 1)* divisor  for item in q]
            n = int(len(points) / 2)
            origin = points[:n]
            target = points[-n:]
            return origin, target

        loops = {}
        for k, line in enumerate(lines):
            origin, target = get_origin_and_targets(line, self.divisor)
            loops.update({k: (origin, target)})

        plt.show()

        with open(self.pickle_file, 'wb') as handle:
            pickle.dump((data, mat), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return loops
