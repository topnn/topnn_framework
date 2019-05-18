import numpy as np
from transformers import Transformer
import plotly.offline as py
import plotly.graph_objs as go
import os
class Visualizer2DLoops(Transformer):

    def __init__(self,  output_filename="2d-scatter-grid-colorscale.html"):
        self._output_filename = output_filename

        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

    def predict(self, content=None):
        """ Does not support predict instead produces visualization of the input contents
        """
        return self.transform(content)

    def transform(self, content=None):
        """Read tfrecords database"""

        trace_training = go.Scatter()
        trace_test = go.Scatter()
        trace_valid = go.Scatter()
        trace_loops = go.Scatter()

        for k, betti_dict in enumerate(content['betti_list']):

            loops = betti_dict['trace']
            betti_name = betti_dict['name']

            if 'nn_predictions_on_test_dataset' in content.keys():

                # first tensor should be ArgMax with network's predictions.
                labels = [item[0] for item in content['nn_predictions_on_test_dataset']['labels']]
                labels = np.array(labels).astype(int)
                labels.shape = (labels.shape[0], )

                samples = np.array(content['nn_predictions_on_test_dataset']['samples'])

                trace_test = go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                        mode='markers',
                                        marker=dict(size=11,
                                        color=labels,  # set color to an array/list of desired values
                                        colorscale='Viridis',  # choose a color scale
                                        opacity=0.2,
                                        line=dict(width=0),
                                        symbol='square'),
                                        name='samples')
            lines = []

            if 'nn_predictions_on_test_dataset' in content.keys():
                samples = np.array(content['nn_predictions_on_test_dataset']['samples'])
            else:
                samples = np.array(content['test_dataset']['samples'])

            colors = ['rgb(67, 104, 31)', 'rgb(255, 180, 3)', 'rgb(129, 255, 3)', 'rgb(3, 214, 255)',
                      'rgb(197, 3, 255)', 'rgb(0,0, 0)', 'rgb(173,218, 69)']

            for k, key in enumerate(loops.keys()):

                 origins, targets = loops[key]
                 lines.extend([{
                    'type': 'line',
                    'x0': samples[origins[i]][0],
                    'y0': samples[origins[i]][1],
                    'x1': samples[targets[i]][0],
                    'y1': samples[targets[i]][1],
                    'line': {
                        'color': colors[k % 7],
                        'width': 7,
                        # 'dash': 'dot',
                    }} for i in range(len(origins))])

            data = [trace_training, trace_test]
            layout = dict(title='Testing of ' + betti_name,
                          width=700, height=700, autosize=False, showlegend= False,
                          shapes=lines)

            fig = go.Figure(data=data, layout=layout)
            py.plot(fig, filename=self._output_filename + betti_name + '.html', auto_open=False)