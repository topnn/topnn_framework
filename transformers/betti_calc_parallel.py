from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
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

class BettiCalcParallel(Transformer):
    def __init__(self, model_path, julia, divisor, neighbors, max_dim):
        self.project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.include_eirene_run = os.path.join(self.project_path, 'julia_include')
        self.betti_dir = os.path.join(model_path, 'betti')
        self.pickle_file =  os.path.join(model_path, 'betti', "workspace_betti.pkl")
        if not os.path.exists(self.betti_dir):
            os.makedirs(self.betti_dir)

        self.verbose = True
        self.number_of_neighbors = int(neighbors)
        self.divisor = int(divisor)
        self.max_dim = int(max_dim)

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

            if X.shape[0] < n_neighbors + 1:
                n_neighbors = X.shape[0]

            nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                    algorithm='ball_tree').fit(X)

            now = print_time('NearestNeighbors - done', now)
            distances, indices = nbrs.kneighbors(X)
            now = print_time('kneighbors - done',now)
            A = nbrs.kneighbors_graph(X).toarray()
            now = print_time('kneighbors_graph - done', now)
            return indices, A

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

        maxdim = self.max_dim
        np.save(mat_file, mat)
        print("fn2 call :", "neihbors_betti_file", neihbors_betti_file, "neihbors_curve_file", neihbors_curve_file, "file_repres_file",  file_repres_file,
              "maxdim", maxdim, "mat_file", mat_file)

        call = ("julia " +  os.path.join(self.include_eirene_run, "include_eirene_run.jl") + ' "' + neihbors_betti_file + '" ' + '"' +neihbors_curve_file+ '" ' +  '"' +file_repres_file + '" '+ str(maxdim) + ' '  '"'+ mat_file + '.npy"' )

        return (call, mat_file + ".npy")