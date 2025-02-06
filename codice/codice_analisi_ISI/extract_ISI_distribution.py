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

def compute_ISI(id_neuroni):

    isi=np.array([])
    id=np.array([])
    for id_neuron in np.nditer(id_neuroni):
        print(id_neuron)
        sp=spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][:]
        isi=np.concatenate((isi,sp[1:] - sp[:-1]))
        id=np.concatenate((id,id_neuron*np.ones(sp.__len__()-1)))
        #isi.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][1:] - spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][:-1])

    return isi,id



current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
parent_path = os.path.dirname(parent_path)

data_net_path=parent_path+"/input_data/data_net/"

filename_PC = data_net_path+"SP_PC_to_SP_PC.hdf5"

f_pyr=h5py.File(filename_PC, "r")
pyr_connection_list=list(f_pyr.keys())

N_pyr=f_pyr[pyr_connection_list[0]][:,0].max()


setting_file = "./configuration.json"
sim_conf = json.load(open('%s' % (setting_file), 'r'))
sim_name=sim_conf["folder_simulation_name"]#"sl9_9-1-9-1"
isi_bin=sim_conf['ISI_bin']
freq_bin=sim_conf['freq_bin']
N_neu_tot=sim_conf['n_of_neurons_in_the_network']
sim_path= parent_path+"/input_data/sim/"+sim_name+"/"
network_results_path=parent_path+"/results/network/"
N_int=N_neu_tot-N_pyr
try:
    os.mkdir(network_results_path)
except FileExistsError:
    pass

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

results_path=results_path+"/ISI_analysis/"

try:
    os.mkdir(results_path)
except FileExistsError:
    pass

list_of_list_spk=h5py.File(sim_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]
fn=np.unique(spk_list[:,0])
try:
    with open(results_path+'info_ISI.pkl', 'rb') as f:
        [ISI,id_n,freq] = pickle.load(f)
except:
    [ISI,id_n]=compute_ISI(fn)
    freq=1000/ISI
    with open(results_path+'info_ISI.pkl', 'wb') as file:  # Python 3: open(..., 'wb')
        pickle.dump([ISI,id_n,freq] , file)


ID_ISI_FREQ=np.vstack((id_n,ISI,freq))

t_final_analysis=spk_list[:,1].max()
t_initial_analysis=0

n_bins=int((t_final_analysis-t_initial_analysis)/isi_bin)
[hist_value, bins] = np.histogram(ISI, n_bins,[t_initial_analysis,t_final_analysis])




fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist_norm.html")


[hist_value, bins] = np.histogram(ISI[id_n<=N_pyr], n_bins,[t_initial_analysis,t_final_analysis])




fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist_pyr.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist_norm_pyr.html")


[hist_value, bins] = np.histogram(ISI[id_n>N_pyr], n_bins,[t_initial_analysis,t_final_analysis])




fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist_int.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"Isi_hist_norm_int.html")




n_bins=int((t_final_analysis-t_initial_analysis)/freq_bin)
[hist_value, bins] = np.histogram(freq, n_bins,[t_initial_analysis,t_final_analysis])


fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist_norm.html")



[hist_value, bins] = np.histogram(freq[id_n<=N_pyr], n_bins,[t_initial_analysis,t_final_analysis])


fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist_pyr.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist_norm_pyr.html")


[hist_value, bins] = np.histogram(freq[id_n>N_pyr], n_bins,[t_initial_analysis,t_final_analysis])


fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist_int.html")



fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value/hist_value.sum(),
            name='isi',
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color="blue",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))

fig.write_html( results_path+"freq_hist_norm_int.html")

isi_und_soglia=ISI<3
tripletta=np.logical_and(np.logical_and(isi_und_soglia[:-1],isi_und_soglia[1:]),id_n[:-1]==id_n[1:])
quadruplette=np.logical_and(np.logical_and(tripletta[:-1],tripletta[1:]),id_n[:-2]==id_n[2:])
#id_neu_with_quad=np.unique(id_n[:-2][quadruplette])
[id_neu_with_quad,n_quad_per_neuron]=np.unique(id_n[:-2][quadruplette],return_counts=True)
id_int_with_quad=id_neu_with_quad[id_neu_with_quad>N_pyr]
n_quad_per_neuron_int=n_quad_per_neuron[id_neu_with_quad>N_pyr]
id_pyr_with_quad=id_neu_with_quad[id_neu_with_quad<=N_pyr]
n_quad_per_neuron_pyr=n_quad_per_neuron[id_neu_with_quad<=N_pyr]
perc_pyr_with_quad=id_pyr_with_quad.__len__()/N_pyr
perc_int_with_quad=id_int_with_quad.__len__()/N_int
