from typing import List, Union
import numpy as np
from transformers import Transformer
import csv
import os
import csv
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing

class TensorsValuesPlotter(Transformer):
    def __init__(self, content_name, output_filename, enable=True):

        self.plots_path = output_filename
        self.content_name = content_name
        self.divisor = 1
        if not os.path.exists(output_filename):
            os.makedirs(output_filename)
        self.enable = enable

    def transform(self, content=None, dim_1=0, dim_2=1, dim_3=2, auto_open=False, normalize=True):

        if type(content) == type({}):

           if self.enable:
                tensor_names = content[self.content_name]['labels_names']
                tensor_values = content[self.content_name]['labels']

                print("plotting those tensors:", tensor_names)
                color_scale = np.array([item[0][0] for item in tensor_values])

                for i in range(len(tensor_names)):
                    flag_2d = False
                    name = tensor_names[i]

                    name = name.replace("/", "-")
                    name = name.replace(":", "-")

                    values = np.array([item[i][0] for item in tensor_values])

                    shape = values.shape
                    if len(shape) == 1:
                        print("warning: tensor %s is a list, will not plot (only 3d plot supported)" % (name))
                        continue

                    if shape[1] > 3:
                        print("warning: tensor %s has dimensions %s, will plot the first 3 dimensions" %(name, shape[1]))

                        if normalize:
                            max_abs_scaler = preprocessing.MaxAbsScaler()
                            values = max_abs_scaler.fit_transform(values)

                        x = values[::self.divisor, dim_1]
                        y = values[::self.divisor, dim_2]
                        z = values[::self.divisor, dim_3]


                    else: # less or equal to 3

                        if shape[1] < 3:
                            print("warning: tensor %s has dimensions %s)" % (name, shape[1]))
                            if normalize:
                                max_abs_scaler = preprocessing.MaxAbsScaler()
                                values = max_abs_scaler.fit_transform(values)

                            x = values[::self.divisor, dim_1]
                            y = values[::self.divisor, dim_2]
                            z = [0] * len(values[::self.divisor, dim_1])
                            flag_2d = True
                        else:
                            if normalize:
                                max_abs_scaler = preprocessing.MaxAbsScaler()
                                values = max_abs_scaler.fit_transform(values)

                            x = values[::self.divisor, dim_1]
                            y = values[::self.divisor, dim_2]
                            z = values[::self.divisor, dim_3]

                    trace_gt = go.Scatter3d(x=x, y=y, z=z,
                                          mode='markers',
                                          marker=dict(size=2,
                                          colorscale='Jet',
                                          color = color_scale,
                                          opacity=0.3,
                                          ),
                                          name=name)

                    data = [trace_gt]

                    layout = dict(title='tensor values for %s' %(name),
                                  width=700, height=700,
                                  showlegend= False,
                                  scene=dict(
                                  xaxis=dict(
                                      nticks=4, range=[-1, 1], ),
                                  yaxis=dict(
                                      nticks=4, range=[-1, 1], ),
                                  zaxis=dict(
                                      nticks=4, range=[-1, 1], ), )
                                  )

                    fig = go.Figure(data=data, layout=layout)
                    file_name = os.path.join(self.plots_path, name +  ".html")
                    py.plot(fig, filename=file_name, auto_open=auto_open)

                    # category plot

                    values_cat1 = np.array([values[i,:] for i in range(len(values)) if color_scale[i] == 1])

                    if len(values_cat1) > 0:
                        if flag_2d:
                            x = values_cat1[::self.divisor, dim_1]
                            y = values_cat1[::self.divisor, dim_2]
                            z = [0] * len(values_cat1[::self.divisor, dim_1])
                        else:
                            x = values_cat1[::self.divisor, dim_1]
                            y = values_cat1[::self.divisor, dim_2]
                            z = values_cat1[::self.divisor, dim_3]

                        trace_gt = go.Scatter3d(x=x, y=y, z=z,
                                              mode='markers',
                                              marker=dict(size=2,
                                              colorscale='Jet',
                                              # color = color_scale,
                                              opacity=0.3,
                                              ),
                                              name=name)

                        data = [trace_gt]

                        layout = dict(title='tensor values for %s' %(name),
                                      width=700, height=700,
                                      showlegend= False,
                                      scene=dict(
                                      xaxis=dict(
                                          nticks=4, range=[-1, 1], ),
                                      yaxis=dict(
                                          nticks=4, range=[-1, 1], ),
                                      zaxis=dict(
                                          nticks=4, range=[-1, 1], ), )
                                      )

                        fig = go.Figure(data=data, layout=layout)
                        file_name = os.path.join(self.plots_path, name + "_cat1_ " + ".html")
                        py.plot(fig, filename=file_name, auto_open=auto_open)


                    values_cat2 =np.array([values[i,:] for i in range(len(values)) if color_scale[i] == 0])

                    if len(values_cat2) > 0:

                        if flag_2d:
                            x = values_cat2[::self.divisor, dim_1]
                            y = values_cat2[::self.divisor, dim_2]
                            z = [0] * len(values_cat1[::self.divisor, dim_1])
                        else:
                            x = values_cat2[::self.divisor, dim_1]
                            y = values_cat2[::self.divisor, dim_2]
                            z = values_cat2[::self.divisor, dim_3]


                        trace_gt = go.Scatter3d(x=x, y=y, z=z,
                                                mode='markers',
                                                marker=dict(size=2,
                                                            colorscale='Jet',
                                                            # color=color_scale,
                                                            opacity=0.3,
                                                            ),
                                                name=name)

                        data = [trace_gt]

                        layout = dict(title='tensor values for %s' % (name),
                                      width=700, height=700,
                                      showlegend=False,
                                      scene=dict(
                                          xaxis=dict(
                                              nticks=4, range=[-1, 1], ),
                                          yaxis=dict(
                                              nticks=4, range=[-1, 1], ),
                                          zaxis=dict(
                                              nticks=4, range=[-1, 1], ), )
                                      )

                        fig = go.Figure(data=data, layout=layout)
                        file_name = os.path.join(self.plots_path, name + "_cat2_ " + ".html")
                        py.plot(fig, filename=file_name, auto_open=auto_open)
           return content
        else:
            return ''