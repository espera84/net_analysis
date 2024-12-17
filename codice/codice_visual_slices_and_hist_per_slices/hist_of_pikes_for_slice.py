import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
import json
import pandas as pd
import plotly.graph_objs as go
import h5py




setting_file="./sim_configuration.json"
sim_conf = json.load(open('%s'%(setting_file), 'r'))
path = sim_conf['Path']
start_time = sim_conf['Time_setting']['Start_time']
simulation_time= sim_conf['Time_setting']['simulation_time']
bin_size=sim_conf['Time_setting']['temporal_bin_size']
automatic_bin_selection=True

plt.ioff()
int_pyr=True

filename_pos="positions.hdf5"
f_pos=h5py.File(filename_pos, "r")

pos_neuron_list=list(f_pos.keys())

neurons_families_list=list(f_pos.keys())
if int_pyr:
    labels = ['Pyr','Int']
    n_pop = 2
else:
    labels=neurons_families_list
    n_pop=neurons_families_list.__len__()

path_results =path+"hist_for_slice/"
path_slice_folder="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/mouse_data/"
list_of_list_spk=[]


try:
    os.mkdir(path_results)
except FileExistsError:
    pass



list_of_list_spk = h5py.File(path + "activity_network.hdf5", "r")
spk_list = list_of_list_spk['spikes'].astype('float64')[:]
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
    ns=pd.read_csv(path_slice_folder+'Slice_'+str(i)+'.csv')
    neuron_slice[i]=np.array(ns['gid'])
    neuron_spk_slice[i]=spk_list[np.in1d(spk_list[:, 0].astype(int),neuron_slice[i]),0]
    spk_sl[i]=neuron_spk_slice[i]=spk_list[np.in1d(spk_list[:, 0].astype(int),neuron_slice[i]),1]
    [hist, bins] = np.histogram(spk_sl[i],100,[start_time,simulation_time])
    hists[i]=hist
    bins_centers=bins[:-1]+(bins[1:]-bins[:-1])/2

    if (i==7):
        points=go.Scatter3d(x=ns['x'],y=ns['y'],z=ns['z'],mode='markers',
                          marker=dict(size=1,
                                      color='blue',
                                      showscale=True, opacity=0.3),)
        fig2 = go.Figure(data=points)
    if (i ==10):
        fig2.add_scatter3d(x=ns['x'], y=ns['y'], z=ns['z'],
                              mode='markers',
                              marker=dict(size=1,
                                          color='red',
                                          showscale=True, opacity=0.3), )
    if (i == 11):
        fig2.add_scatter3d(x=ns['x'],y=ns['y'],z=ns['z'],
                          mode='markers',
                          marker=dict(size=1,
                                      color='green',
                                      showscale=True, opacity=0.3),)
    if (i == 12):
        fig2.add_scatter3d(x=ns['x'], y=ns['y'], z=ns['z'],
                              mode='markers',
                              marker=dict(size=1,
                                          color='yellow',
                                          showscale=True, opacity=0.3),)

    plt.plot(bins_centers, hist, label='slice_' + str(i),color=color.colors[i])


plt.legend()
plt.show()
plt.savefig(path_results+'hist_for_slice.png')
fig2.add_scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                  y=f_pos[pos_neuron_list[10]][0::10, 2],
                  z=f_pos[pos_neuron_list[10]][0::10, 3],
                  mode='markers',
                  marker=dict(size=1,
                              color='gray',
                              showscale=True, opacity=0.3), )

fig2.show()

ns=pd.read_csv(path_slice_folder+'gid_stim_CA3_1600.csv')
points=go.Scatter3d(x=ns['x'],y=ns['y'],z=ns['z'],mode='markers',
                          marker=dict(size=1,
                                      color='blue',
                                      showscale=True, opacity=0.3),)
fig2 = go.Figure(data=points)
fig2.add_scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                  y=f_pos[pos_neuron_list[10]][0::10, 2],
                  z=f_pos[pos_neuron_list[10]][0::10, 3],
                  mode='markers',
                  marker=dict(size=1,
                              color='gray',
                              showscale=True, opacity=0.3)
                  , )

fig2.show()

