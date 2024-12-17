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

filename_pos="positions.hdf5"
#with  as f:
f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())


type_to_plot="int"
[n_neu_in_sub_pyr,sub_net_pos_pyr,sub_net_pyr,n_neu_in_sub_int,sub_net_pos_int,sub_net_int]=compute_sub_net( 20,34,34)

sub_net_pos_pyr=np.array(sub_net_pos_pyr)
fig =px.scatter_3d(x=sub_net_pos_pyr[:,0], y=sub_net_pos_pyr[:, 1], z=sub_net_pos_pyr[:, 2],color=n_neu_in_sub_pyr,opacity=0.7,size=n_neu_in_sub_pyr)
#fig.show()

sub_net_pos_int=np.array(sub_net_pos_int)
fig =px.scatter_3d(x=sub_net_pos_int[:,0], y=sub_net_pos_int[:, 1], z=sub_net_pos_int[:, 2],color=n_neu_in_sub_int,opacity=0.7,size=n_neu_in_sub_int)
#fig.show()


path_slice_folder="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/mouse_data/"
list_of_list_spk=[]


setting_file="./sim_configuration.json"
sim_conf = json.load(open('%s'%(setting_file), 'r'))
path = sim_conf['Path']
start_time = sim_conf['Time_setting']['Start_time']
simulation_time= sim_conf['Time_setting']['simulation_time']
bin_size=sim_conf['Time_setting']['temporal_bin_size']


neuron_slice=np.empty((19),dtype=object)
neuron_spk_slice=np.empty((19),dtype=object)
hists=np.empty((19),dtype=object)
spk_sl=np.empty((19),dtype=object)
i=1
ns=pd.read_csv(path_slice_folder+'Slice_'+str(i)+'.csv', header=None)
plt.figure()

color = cm.tab20# cm.rainbow(np.linspace(0, 1, 20))
for i in range(19):

    neuron_slice[i]=[]
    neuron_slice[i]=pd.read_csv(path_slice_folder+'Slice_'+str(i)+'.csv')

color = cm.tab20# cm

slice_to_plot=[8,11]
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)
if type_to_plot=="pyr":
    app.layout = html.Div([
        html.H4('concentration'),
        dcc.Graph(id="graph"),
        html.P("number of neurons:"),
        dcc.RangeSlider(
            id='range-slider',
            min=0, max=650, step=1,
            marks={0: '0', 650: '650'},
            value=[1, 650]
        ),
    ])
else:
    app.layout = html.Div([
        html.H4('concentration'),
        dcc.Graph(id="graph"),
        html.P("number of neurons:"),
        dcc.RangeSlider(
            id='range-slider',
            min=0, max=100, step=1,
            marks={0: '0', 100: '100'},
            value=[1, 100]
        ),
    ])

@app.callback(
    Output("graph", "figure"),
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    #df = px.data.iris() # replace with your own data source
    global type_to_plot,f_pos,pos_neuron_list,slice_to_plot
    low, high = slider_range


    if type_to_plot=="pyr":
        mask = np.logical_and(np.array(n_neu_in_sub_pyr)>low,np.array(n_neu_in_sub_pyr)<high)

        fig = px.scatter_3d(x=sub_net_pos_pyr[mask, 0], y=sub_net_pos_pyr[mask, 1], z=sub_net_pos_pyr[mask, 2],
                        color=np.array(n_neu_in_sub_pyr)[mask], opacity=0.7, size=np.array(n_neu_in_sub_pyr)[mask], size_max=18,width=2000, height=800,title="pyramidal concentration")
    else:
        mask = np.logical_and(np.array(n_neu_in_sub_int) > low, np.array(n_neu_in_sub_int) < high)

        fig = px.scatter_3d(x=sub_net_pos_int[mask, 0], y=sub_net_pos_int[mask, 1], z=sub_net_pos_int[mask, 2],
                        color=np.array(n_neu_in_sub_int)[mask], opacity=0.7, size=np.array(n_neu_in_sub_int)[mask],
                        size_max=18, width=2000, height=800, title="interneurons concentration")


    fig.add_scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                      y=f_pos[pos_neuron_list[10]][0::10, 2],
                      z=f_pos[pos_neuron_list[10]][0::10, 3],
                      mode='markers',
                      marker=dict(size=1,
                                  color='gray',
                                  showscale=True, opacity=0.3)
                      ,)
    for i in slice_to_plot:

        fig.add_scatter3d(x=neuron_slice[i]['x'], y=neuron_slice[i]['y'], z=neuron_slice[i]['z'],
                      mode='markers',
                      marker=dict(size=1,
                                  color="yellow",#color.colors[i],
                                  showscale=True, opacity=0.05), )
    return fig

print("loadind info connection")
with open('connections_lists.pkl', 'rb') as f:
    [in_conn, out_conn] = pickle.load(f)
print("info connection loaded")
def find_stimulation_neurons3(ind):

    global in_conn


    out = in_conn[ind]
    out2=[]
    for k in range(out.__len__()):
                #print(k)
        out2 = np.unique(out2 + out[k]).tolist()
        id_neu_che_potenzialmente_stimolano = np.unique(out2)


        return id_neu_che_potenzialmente_stimolano
    else:
        return np.empty(0)


if type_to_plot=="pyr":
    app.run_server(port=8050)
else:
    app.run_server(port=8052)

