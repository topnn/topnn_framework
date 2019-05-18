from typing import List, Union
import numpy as np
from transformers import Transformer
import csv
import os
import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

class ExcelReader(Transformer):
    def __init__(self, input_filename="saved_dataset.csv"):
        self._input_filename = input_filename
        self.step2load = '1000'

    def transform(self, content=None):
        if os.path.isdir(self._input_filename):
            print("\nExcel Reader: given  directory: " + self._input_filename)
            files = sorted(glob.glob(self._input_filename + '/*.csv'))
            print("\nExcel Reader found: ")
            print(files, sep='\n')
        else:
            files = [self._input_filename]


        dictforplot = {}
        names =['denseMatMul', 'denseBiasAdd', 'LeakyRelu',
                'dense_1MatMul' , 'dense_1BiasAdd', 'LeakyRelu_1',
                'dense_2MatMul', 'dense_2BiasAdd', 'LeakyRelu_2',
                'dense_3MatMul', 'dense_3BiasAdd', 'LeakyRelu_3',
                'dense_4MatMul', 'dense_4BiasAdd', 'LeakyRelu_4',
                'dense_5MatMul', 'dense_5BiasAdd', 'LeakyRelu_5',
                'dense_6MatMul', 'dense_6BiasAdd', 'LeakyRelu_6',
                'dense_7MatMul', 'dense_7BiasAdd', 'LeakyRelu_7',
                'dense_8MatMul', 'dense_8BiasAdd', 'LeakyRelu_8' ]
        cap = cv2.VideoCapture(0)  # video source: webcam
        fourcc = cv2.VideoWriter.fourcc(*'XVID')  # record format xvid
        out = cv2.VideoWriter('output.avi', fourcc, 10, (640, 480))
        regex = re.compile('(\S+)_(\S+\d)_(\S+)_*(\d*)-(\d)\.csv')
        for file in files:
            f = os.path.basename(file)
            result = regex.match(f)
            print(result.group(0))
            print(result.group(1))
            print(result.group(2))
            print(result.group(3))
            print(result.group(4))
            print(result.group(5))
            if not (result.group(2) == 'cat2' and self.step2load in result.group(1)):
                continue
            with open(file) as csv_file:
                rows = []
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    print(row)
                    rows.append(row)
            rows= np.asarray(rows,dtype=float)
            dictforplot.update({result.group(3) : rows})

        for name in names:
            rows = dictforplot[name]
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(rows[0,:], rows[1,:], rows[2,:], s=0.1)
            ax.set_axis_off()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            azimuths = [0, 30 , 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            elevations = [0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30]
            for i, azimuth in enumerate(azimuths):
                el = elevations[2]
                ax.view_init(elev=el, azim=azimuth)
                plt.savefig('last.png')
                img = cv2.imread("last.png")
                cv2.putText(img=img,text= 'cat-1:   ' + name , org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(200, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                out.write(img)
        pass
        out.release()



    def predict(self, content=None):
        """saver done't support predict instead writes content on the disk"""

        return self.transform(content)
