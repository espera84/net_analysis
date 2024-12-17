from network_utility import compute_sub_net
import plotly.express as px
import numpy as np
import h5py
import json
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pickle
import os

filename_pos="positions.hdf5"
#with  as f:
f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())

first_neuron = 0
last_neuron = 2000
sigla="sl9"
t_initial_analysis = 30000
t_final_analysis = 40000
interval_dim=1
with open('phase_and_freq_neurons_from_' + str(first_neuron) + '_' + str(last_neuron) + '_' + sigla + '_' + str(t_initial_analysis) + '_' + str(t_final_analysis) + '_' + str(
        interval_dim) + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [fr, rad,neurons_selected] = pickle.load(f)
color = cm.tab20# cm



posizioni_neuroni=f_pos[pos_neuron_list[0]][:]
#print(str(f_pos[pos_neuron_list[0]][:].__len__())+" "+str(posizioni_neuroni.__len__()))
for j in range(1,len(pos_neuron_list)):
    #print(str(f_pos[pos_neuron_list[j]][:].__len__())+" "+str(posizioni_neuroni.__len__()))
    posizioni_neuroni = np.concatenate((posizioni_neuroni, f_pos[pos_neuron_list[j]][:]))


points = go.Scatter3d(x=posizioni_neuroni[0::10, 1],
                          y=posizioni_neuroni[0::10, 2],
                          z=posizioni_neuroni[0::10, 3],
                          name='network subsampling ',
                          mode='markers',
                          marker=dict(size=1,
                                      color='gray',
                                      # showscale=False,
                                      opacity=0.3),
                          )
camera = dict(eye=dict(x=2, y=0, z=0))
layout = go.Layout(margin=dict(l=0,
                                   r=0,
                                   b=0,
                                   t=10),
                       scene_camera=camera)

fig = go.Figure(data=points, layout=layout)

from dash import Dash, dcc, html, Input, Output
from dash.dependencies import State
import plotly.express as px

app = Dash(__name__)




app.layout = html.Div([
    html.H4('concentration'),
    dcc.Graph(id="graph",figure=fig),
    html.P("freq:"),
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=30, step=0.1,
        marks={0: '0', 30: '30'},
        value=[0, 30]
    ),

    dcc.RangeSlider(
        id='range-slider2',
        min=-np.pi, max=np.pi, step=0.1,
        marks={-np.pi: '-P', np.pi: 'P'},
        value=[-np.pi, np.pi]
    ),

    dcc.Store(id='store-camera', data=None),

    #dcc.Graph(id='3d-plot', figure=fig),
    html.Div(id='camera-info', style={'padding': '20px', 'fontSize': '18px'})

])

@app.callback(
    #Output("graph", "figure"),
    #Output('3d-plot', 'figure'),
    Output('camera-info', 'children'),
    Input("range-slider", "value"),
    Input("range-slider2", "value"),
    Input('graph', 'relayoutData')
)
def update_bar_chart(slider_range,slider_range2,relayoutData):
    #df = px.data.iris() # replace with your own data source
    global f_pos,pos_neuron_list,neurons_selected
    low_fr, high_fr = slider_range
    low_ph,high_ph =  slider_range2
    print(slider_range)
    print(slider_range2)
    print(relayoutData)
    # if stored_camera != None:
    #     print("oooooo000000000000000")
    #     print(stored_camera)
    #     camera=stored_camera['scene']['camera']
    # else:
    #     camera = dict(eye=dict(x=2, y=0, z=0))


    fig = px.scatter_3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                         y=f_pos[pos_neuron_list[10]][0::10, 2],
                         z=f_pos[pos_neuron_list[10]][0::10, 3],
                         size=np.ones(f_pos[pos_neuron_list[10]][0::10, 3].__len__()) * 0.3,
                         opacity=0.5, size_max=1)



    neuron_to_plt = np.in1d(posizioni_neuroni[:, 0], np.array(neurons_selected)[np.in1d(neurons_selected, np.where(np.logical_and(np.logical_and(fr > low_fr, fr <= high_fr),np.logical_and(rad > low_ph, rad <= high_ph))))])

    fig=go.Figure()
    fig.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                       y=posizioni_neuroni[neuron_to_plt, 2],
                       z=posizioni_neuroni[neuron_to_plt, 3],
                       name='frequency ' + str(low_fr) + ' ' + str(high_fr)+' phase ' + str(low_ph) + ' ' + str(high_ph),
                       text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                       mode='markers',
                       marker=dict(size=4,
                                   color=rad[np.where(np.logical_and(np.logical_and(fr > low_fr, fr <= high_fr),np.logical_and(rad > low_ph, rad <= high_ph)))],
                                   # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),,
                                   showscale=True,
                                   opacity=1,
                                   colorscale='Phase'),
                       )
    # if type_to_plot == "pyr":
    #     fig.write_html(path_res + "concentration_map_pyr.html")
    # else:
    #     fig.write_html(path_res + "concentration_map_int.html")

    # if 'scene.camera' in relayoutData:
    #     camera = relayoutData['scene.camera']
    #     camera_info = f"Posizione Camera: x={camera['eye']['x']}, y={camera['eye']['y']}, z={camera['eye']['z']}"
    #
    #
    #     return relayoutData,camera_info

    return relayoutData,"Muovi la vista per aggiornare la posizione della camera..."



app.run_server(port=8062)
