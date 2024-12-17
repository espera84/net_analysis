import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pickle
import os
import h5py
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
import sys
import json



if sys.argv.__len__()>2:
    work_path = sys.argv[1]
    sigla = sys.argv[2]
    setting_file = sys.argv[3]
    sim_conf = json.load(open('%s' % (setting_file), 'r'))

else:
    work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
    work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
    #work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
    sigla = "sl9"
    sigla="ls5"
    #sigla = "bkg_5hz"
    setting_file = "./configuration.json"
    sim_conf = json.load(open('%s' % (setting_file), 'r'))
t_initial_analysis = sim_conf['start_time']
t_final_analysis = sim_conf['end_time']  # 0#5000#
interval_dim = sim_conf['bins_dimension']
n_workers = -1

n_neuroni=288027
print(work_path)

list_of_list_spk=h5py.File(work_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]
fn=np.unique(spk_list[:,0])

iniz_intervals=[]
fine_intervals=[]

t_iniz=t_initial_analysis
t_fin=t_initial_analysis+interval_dim

while(t_fin<=t_final_analysis):
    iniz_intervals.append(float(t_iniz))
    fine_intervals.append(float(t_fin))
    t_iniz=t_fin
    t_fin=t_fin+interval_dim
fine_intervals=np.array(fine_intervals)
iniz_intervals=np.array(iniz_intervals)
time=(iniz_intervals+fine_intervals)/2

[fn, n_spk_fn]=np.unique(spk_list[:,0], return_counts=True)
n_spk_tutti=np.zeros(n_neuroni+1)
n_spk_tutti[fn.astype(int)]=n_spk_fn

filename_pos="positions.hdf5"
f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())

posizioni=[]
for i in  range(len(pos_neuron_list)):
    posizioni.append(f_pos[pos_neuron_list[i]][:])

posizioni_neuroni=posizioni[0]
for j in range(1,posizioni.__len__()):
    posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))

ordine_neu=posizioni_neuroni[:,0].argsort()
posizioni_neu_ord=posizioni_neuroni[ordine_neu,:].copy()

massimo =2000
n_spk_tutti[n_spk_tutti[:]>massimo]=massimo

layout = go.Layout(title='number of spikes',
                       margin=dict(l=0,
                                   r=0,
                                   b=0,
                                   t=0)
                       )
fig2 = go.Figure(layout=layout)

fig2.add_scatter3d(x = np.array(posizioni_neu_ord)[:,1],
                    y = np.array(posizioni_neu_ord)[:,2],
                    z = np.array(posizioni_neu_ord)[:,3],
                    mode = 'markers',
                    marker = dict( size = 1,
                                   colorscale='pinkyl',
                                   color=n_spk_tutti[1:],#np.random.rand(1)[0]n,#color_discrete_sequence[n],#n/10,
                                   showscale=True)
                    )  # ,showscale=True), )
fig2.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
            )

fig2.show()