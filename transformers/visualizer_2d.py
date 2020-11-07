import numpy as np
from transformers import Transformer
import plotly.offline as py
import plotly.graph_objs as go
import os
class Visualizer2D(Transformer):

    def __init__(self,  output_filename="2d-scatter-grid-colorscale.html", mode='2D', enable=True):
        self._output_filename = output_filename
        self.mode = mode
        self.enable = enable

        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

    def predict(self, content=None):
        """ Does not support predict instead produces visualization of the input contents
        """
        return self.transform(content)

    def transform(self, content=None):
        """Read tfrecords database"""
        if type(content) == type({}):
            if self.enable:
                if self.mode == '2D':
                    trace_training = go.Scatter()
                    trace_test = go.Scatter()
                    trace_valid = go.Scatter()
                    trace_loops = go.Scatter()
                else:
                    trace_training = go.Scatter3d()
                    trace_test = go.Scatter3d()
                    trace_valid = go.Scatter3d()
                    trace_loops = go.Scatter3d()

                if 'training_dataset' in content.keys():

                    labels = np.array(content['training_dataset']['labels']).astype(int)
                    labels.shape = (labels.shape[0], )
                    samples = np.array(content['training_dataset']['samples'])
                    if self.mode == '2D':
                        trace_training = go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                              mode='markers',
                                              marker=dict(size=5,
                                              # marker=dict(size=1,
                                              color=labels,  # set color to an array/list of desired values
                                              colorscale='Jet',  # choose a color scale
                                              opacity=0.9,
                                              #line=dict(color='rgb(231, 99, 250)', width=0.5)
                                              ),
                                              name='Dataset')
                    else:
                        trace_training = go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                                              mode='markers',
                                              marker=dict(size=5,
                                              # marker=dict(size=1,
                                              color=labels,  # set color to an array/list of desired values
                                              colorscale='Jet',  # choose a color scale
                                              opacity=0,
                                              #line=dict(color='rgb(231, 99, 250)', width=0.5)
                                              ),
                                              name='Dataset')
                if 'nn_predictions_on_test_dataset' in content.keys():

                    # first tensor should be ArgMax with network's predictions.
                    labels = [item[0] for item in content['nn_predictions_on_test_dataset']['labels']]
                    labels = np.array(labels).astype(int)
                    labels.shape = (labels.shape[0], )

                    samples = np.array(content['nn_predictions_on_test_dataset']['samples'])

                    if self.mode == '2D':
                        trace_test = go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                                mode='markers',
                                                marker=dict(size=11,
                                                # marker=dict(size=1,
                                                color=labels,  # set color to an array/list of desired values
                                                colorscale='Viridis',  # choose a color scale
                                                opacity=0.2,
                                                line=dict(width=0),
                                                symbol='square'),
                                                name='samples')
                    else:
                        trace_test = go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                                                mode='markers',
                                                marker=dict(size=11,
                                                # marker=dict(size=1,
                                                color=labels,  # set color to an array/list of desired values
                                                colorscale='Viridis',  # choose a color scale
                                                opacity=0.01,
                                                line=dict(width=0),
                                                symbol='square'),
                                                name='samples')

                if 'nn_predictions_on_validation_dataset' in content.keys():

                    # first tensor should be ArgMax with network's predictions.
                    labels = [item[0] for item in content['nn_predictions_on_validation_dataset']['labels']]
                    labels = np.array(labels).astype(int)
                    labels.shape = (labels.shape[0], )

                    samples = np.array(content['nn_predictions_on_validation_dataset']['samples'])

                    if self.mode == '2D':

                        trace_valid = go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                                mode='markers',
                                                marker=dict(size=11,
                                                # marker=dict(size=1,
                                                color=labels,  # set color to an array/list of desired values
                                                colorscale='Viridis',  # choose a color scale
                                                opacity=0.2,
                                                line=dict(width=0),
                                                symbol='square'),
                                                name='samples')
                    else:
                        trace_valid =  go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                                                mode='markers',
                                                marker=dict(size=11,
                                                # marker=dict(size=1,
                                                color=labels,  # set color to an array/list of desired values
                                                colorscale='Viridis',  # choose a color scale
                                                opacity=0.01,
                                                line=dict(width=0),
                                                symbol='square'),
                                                name='samples')

                data = [trace_training, trace_test, trace_valid]
                layout = dict(title='Data Visualization',
                              width=900, height=900, autosize=False, showlegend= False)

                fig = go.Figure(data=data, layout=layout)
                py.plot(fig, filename=self._output_filename, auto_open=False)

            return content
        else:
            return ''
