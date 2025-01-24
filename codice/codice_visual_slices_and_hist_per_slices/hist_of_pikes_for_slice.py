import numpy as np
from matplotlib.pyplot import cm
import os
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import h5py




setting_file="./sim_configuration.json"
sim_conf = json.load(open('%s'%(setting_file), 'r'))
start_time = sim_conf['Time_setting']['start_time']
end_time= sim_conf['Time_setting']['end_time']
bin_size=sim_conf['Time_setting']['temporal_bin_size']
n_bin=int((end_time-start_time)/bin_size)
sim_name=sim_conf["folder_simulation_name"]
automatic_bin_selection=True

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
parent_path = os.path.dirname(parent_path)
sim_path= parent_path+"/input_data/sim/"+sim_name+"/"

#sim_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_Giulia_17_10/pruning1_bkg_5hz_5151/"
data_net_path=parent_path+"/input_data/data_net/"
results_path=parent_path+"/results/sim/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass

results_path=results_path+sim_name+"/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass

results_path=results_path+"hist_slices_/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass


filename_in = data_net_path+"connections_inh.hdf5"
filename_PC = data_net_path+"SP_PC_to_SP_PC.hdf5"
filename_pos=data_net_path+"positions.hdf5"

f_pos=h5py.File(filename_pos, "r")

pos_neuron_list=list(f_pos.keys())

neurons_families_list=list(f_pos.keys())

mod="pyr"
n_pyr=261843

list_of_list_spk = h5py.File(sim_path + "activity_network.hdf5", "r")
spk_list = list_of_list_spk['spikes'].astype('float64')[:]
neuron_slice=np.empty((19),dtype=object)
neuron_spk_slice=np.empty((19),dtype=object)
hists=np.empty((19),dtype=object)
spk_sl=np.empty((19),dtype=object)
n_neu_firing_slice=np.zeros([19,n_bin])
n_neu_firing_slice_norm=np.zeros([19,n_bin])
n_neu_firing_slice_norm_neu=np.zeros([19,n_bin])


color = cm.tab20# cm.rainbow(np.linspace(0, 1, 20))
first_slice=True
slices_perc_excluded=0
for i in range(19):

    neuron_slice[i]=[]
    ns=pd.read_csv(data_net_path+'Slice_'+str(i)+'.csv')

    if mod=="pyr":
        neuron_selected = np.logical_and(np.array(ns['x']) >= (np.array(ns['x']).min() + (
                    np.array(ns['x']).max() - np.array(ns['x']).min()) * slices_perc_excluded),np.array(ns['gid'])<n_pyr)
        neuron_slice[i] = np.array(ns['gid'])[neuron_selected]

    if mod=="int":
        neuron_selected = np.logical_and(np.array(ns['x']) >= (np.array(ns['x']).min() + (
                np.array(ns['x']).max() - np.array(ns['x']).min()) * slices_perc_excluded), np.array(ns['gid']) > n_pyr)
        neuron_slice[i] = np.array(ns['gid'])[neuron_selected]

    if mod=="all":
        neuron_selected = np.array(ns['x']) >= (np.array(ns['x']).min() + (
                    np.array(ns['x']).max() - np.array(ns['x']).min()) * slices_perc_excluded)
        neuron_slice[i] = np.array(ns['gid'])[neuron_selected]  # np.array(ns['gid'])
    neuron_spk_slice[i]=spk_list[np.in1d(spk_list[:, 0].astype(int),neuron_slice[i]),0]
    spk_sl[i]=spk_list[np.in1d(spk_list[:, 0].astype(int),neuron_slice[i]),1]
    for k in range(n_bin):
        n_neu_firing_slice[i][k]=np.unique(neuron_spk_slice[i][np.logical_and(spk_sl[i] > start_time+ k*bin_size, spk_sl[i] < (start_time + (k+1)*bin_size))]).__len__()
    [hist, bins] = np.histogram(spk_sl[i],n_bin,[start_time,end_time])
    hists[i]=hist
    bins_centers=bins[:-1]+(bins[1:]-bins[:-1])/2

    if first_slice:
        first_slice = False
        fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist,
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))
        if hist.sum()>0:
            hist_norm=hist/hist.sum()
        else:
            hist_norm = hist
        fig_norm = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_norm,
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

        fig_n_neu = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))
        if n_neu_firing_slice[i].sum()>0:
            n_neu_firing_slice_norm[i]=n_neu_firing_slice[i]/n_neu_firing_slice[i].sum()
            n_neu_firing_slice_norm_neu[i] = n_neu_firing_slice[i] / neuron_slice[i].__len__()
        else:
            n_neu_firing_slice_norm[i] = n_neu_firing_slice[i]
        fig_n_neu_norm = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice_norm[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

        fig_n_neu_norm_neu = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice_norm_neu[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))



        points = go.Scatter3d(x=ns['x'][neuron_selected], y=ns['y'][neuron_selected], z=ns['z'][neuron_selected], mode='markers',
                              marker=dict(size=1,
                                          color='blue',
                                          showscale=False, opacity=0.3), )
        fig_slice = go.Figure(data=points)
    else:
        fig.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist,
            name='slice_' + str(i),
            # str(id_center[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )
        if hist.sum()>0:
            hist_norm=hist/hist.sum()
        else:
            hist_norm = hist
        fig_norm.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_norm,
            name='slice_' + str(i),
                # str(id_center[i]),
                # mode='markers',
            marker=dict(
                    # symbol=sim,
                    # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                    # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

        fig_n_neu.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        )
        if n_neu_firing_slice[i].sum() > 0:
            n_neu_firing_slice_norm[i] = n_neu_firing_slice[i] / n_neu_firing_slice[i].sum()
            n_neu_firing_slice_norm_neu[i] = n_neu_firing_slice[i] / neuron_slice[i].__len__()
        else:
            n_neu_firing_slice_norm[i] = n_neu_firing_slice[i]
        fig_n_neu_norm.add_scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice_norm[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        )

        fig_n_neu_norm_neu.add_scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:] + bins[:-1]) / 2,
            y=n_neu_firing_slice_norm_neu[i],
            name='slice_' + str(i),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color.colors[i],  # color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        )



        fig_slice.add_scatter3d(x=ns['x'][neuron_selected], y=ns['y'][neuron_selected], z=ns['z'][neuron_selected],
                           mode='markers',
                           marker=dict(size=1,
                                       color=px.colors.qualitative.Alphabet[i],
                                       showscale=False, opacity=0.3), )


    fig.write_html(results_path + "hist_firing_slices_bin_size_"+str(bin_size)+"_perc_"+str(1-slices_perc_excluded)+mod+".html")
    fig_norm.write_html(results_path + "hist_norm_firing_slices_bin_size_"+str(bin_size)+"_perc_"+str(1-slices_perc_excluded)+mod+".html")
    fig_n_neu.write_html(results_path + "n_neu_firing_slices_bin_size_" + str(bin_size) + "_perc_" + str(1 - slices_perc_excluded) +mod+ ".html")
    fig_n_neu_norm.write_html(results_path + "n_neu_firing_norm_slices_bin_size_" + str(bin_size) + "_perc_" + str(1 - slices_perc_excluded) +mod+ ".html")
    fig_n_neu_norm_neu.write_html(results_path + "n_neu_firing_norm_su_tot_neu_slices_bin_size_" + str(bin_size) + "_perc_" + str(1 - slices_perc_excluded) + mod + ".html")
    fig_slice.write_html(results_path + "slices_visualization_perc_"+str(1-slices_perc_excluded)+".html")



