import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red
from gini_WB import compute_Theil_W_B,compute_Theil
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import h5py
import csv
import pandas as pd

filename_in = "connections_inh.hdf5"
filename_PC = "SP_PC_to_SP_PC.hdf5"
filename_pos="positions.hdf5"
#with  as f:
f_in=h5py.File(filename_in, "r")
in_connection_list=list(f_in.keys())

f_pyr=h5py.File(filename_PC, "r")
pyr_connection_list=list(f_pyr.keys())

f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())

work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/neuroni_alta_attivabilita/"

n=8
list_of_neuron = pd.read_csv(work_path+"altaAttivabilita"+str(n)+"_id.csv", header=None)
#spk_list=np.array(list_of_list_spk)
aux=np.array(list_of_neuron[0])
posizioni=[]
for i in aux:
    #print(i)
    posizioni.append(f_pos[pos_neuron_list[10]][i-1,1:])


points = go.Scatter3d( x = np.array(posizioni)[:,0],
                       y = np.array(posizioni)[:,1],
                       z = np.array(posizioni)[:,2],
                       mode = 'markers',
                       marker = dict( size = 1,
                                      color='darkred',
                                      showscale=True)
                     )


layout = go.Layout(margin = dict( l = 0,
                                  r = 0,
                                  b = 0,
                                  t = 0)
                  )
fig = go.Figure(data=points,layout=layout)
fig.write_html(work_path +"scatter_"+str(n)+".html")



points = go.Scatter3d( x = f_pos[pos_neuron_list[10]][:,1],
                       y = f_pos[pos_neuron_list[10]][:,2],
                       z = f_pos[pos_neuron_list[10]][:,3],
                       mode = 'markers',
                       marker = dict( size = 1,
                                      color='gray',
                                      showscale=True,opacity=0.1)
,

                     )




layout = go.Layout(margin = dict( l = 0,
                                  r = 0,
                                  b = 0,
                                  t = 0)
                  )
fig2 = go.Figure(data=points,layout=layout)

fig2.add_scatter3d(x = np.array(posizioni)[:,0],
                       y = np.array(posizioni)[:,1],
                       z = np.array(posizioni)[:,2],
                       mode = 'markers',
                       marker = dict( size = 1,
                                      color='darkred',
                                      showscale=True)
                  )  # ,showscale=True), )

fig2.write_html(work_path +"scatter2_"+str(n)+".html")

