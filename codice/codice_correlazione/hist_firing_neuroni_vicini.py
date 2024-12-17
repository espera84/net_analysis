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
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
import sys
import json
import time
from numpy import random

def stim_neurons(neurons):
    #neuron_stim_pyr = np.empty(indici_cl.__len__(), dtype=object)
    #neuron_stim_int = np.empty(indici_cl.__len__(), dtype=object)
    n_stim_pyr =0
    n_stim_int =0

    aux = f_pyr[pyr_connection_list[0]][
            np.isin(f_pyr[pyr_connection_list[0]][:, 1], neurons), 0]
    n_stim_pyr = n_stim_pyr+ aux.__len__()
    neuron_stim_pyr = np.unique(aux)

    for j in range(in_connection_list.__len__()):
        if j == 0:
            neuron_stim_int = []
        if in_connection_list[j][:5] == "SP_PC":
            aux = f_in[in_connection_list[j]][np.isin(f_in[in_connection_list[j]][:, 1], neurons), 0]
            n_stim_pyr= n_stim_pyr+ aux.__len__()
            neuron_stim_pyr= np.concatenate((neuron_stim_pyr, np.unique(aux)))
        else:
            aux = f_in[in_connection_list[j]][np.isin(f_in[in_connection_list[j]][:, 1], neurons), 0]
            n_stim_int = n_stim_int+ aux.__len__()
            neuron_stim_int = np.concatenate((neuron_stim_int, np.unique(aux)))

    neuron_stim_pyr = np.unique(neuron_stim_pyr)
    neuron_stim_int = np.unique(neuron_stim_int)
    return neuron_stim_pyr, neuron_stim_int,n_stim_pyr,n_stim_int


def fourier_analysis(signal):
    # Calcola la Trasformata Discreta di Fourier (DFT) per segnali reali usando rfft
    dft = np.fft.rfft(signal)

    # Calcola le frequenze associate (solo la parte positiva)
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=(t_final_analysis-t_initial_analysis)/(1000*signal.__len__()))  # Frequenze in Hz

    # Calcoliamo l'ampiezza (modulo della DFT)
    amplitude = np.abs(dft)

    # Calcoliamo le fasi (in radianti)
    phase = np.angle(dft)

    return freq, amplitude, phase

def save_raster_all2(list_id_neuroni):

    global t_initial_analysis,t_final_analysis
    spk = []

    color = np.array([])
    plt.figure()
    pos_n=1
    k=0
    for j in range(list_id_neuroni.__len__()):
        id_neuroni=list_id_neuroni[j]
        first_neuron=True
        print("lista", j)
        print("lung lista",id_neuroni.__len__())
        #spk_id=spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0],1]
        spk_id=np.vstack((spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0], 1],spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0], 0]))
        spk_id_interval=spk_id[:,np.logical_and(spk_id[0,:]>t_initial_analysis , spk_id[0,:]<t_final_analysis)]
        for i in range(list_id_neuroni[j].__len__()):
            spk_id_interval[1,np.where(spk_id_interval[1,:]==list_id_neuroni[j][i])]=k+i
        k=k+i+1
        color=np.concatenate((color,np.ones(spk_id_interval[0,:].__len__())*j))
            #plt.scatter(spk_list[pos_spk_id, 1][0],i*np.ones(spk_list[pos_spk_id, 1][0].__len__()))
            #spk.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][np.logical_and(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]>t_initial_analysis , spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]<t_final_analysis)])
        spk.append(spk_id_interval)


    for i in range(spk.__len__()):
        if i == 0:
            #cl_spk = np.vstack((np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__()))), np.ones(spk[i].__len__()) * i))
            cl_spk = spk[i]
            #cl_spk = np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))
        else:
            # print(np.ones(spk[i].__len__()))
            #cl_spk = np.concatenate((cl_spk, np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))), axis=1)
            cl_spk = np.concatenate((cl_spk,spk[i]) , axis=1)
        pos_n = pos_n + spk[i].__len__()

    k=0
    #fig = px.scatter(x=cl_spk[0,:], y=cl_spk[1,:],color=color)
    #fig.write_image(results_sub_path + "r_cc.svg")
    #fig.show()

    #
    # plt.eventplot(spk, linelengths=0.7)
    # plt.legend(loc="upper right")
    # plt.title('raster componenti connesse')
    # plt.xlabel("time (ms)")
    # plt.ylabel("neuron")
    # plt.savefig(results_sub_path + "r_cc.svg")
    # plt.clf()
    return cl_spk,color


def calcola_neuroni_vicini(i_iniz,inter):

    global f_pos,pos_neuron_list,posizioni_centri,distanza_massima
    for i in range(i_iniz,i_iniz+inter):
        print(i)
        print('calc dist')
        distanze_from_pyr=f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i]
        print('calc comp dist')
        is_comp_dist_min_dist_max = distanze_from_pyr < distanza_massima
        print('calc is near')
        is_neuron_near=np.logical_and(is_comp_dist_min_dist_max[:,0],is_comp_dist_min_dist_max[:,1],is_comp_dist_min_dist_max[:,2])
        print('add neuron')
        neurons_near_to_center[i]=[]
        neurons_near_to_center[i].append(f_pos[pos_neuron_list[10]][is_neuron_near,0])
        print('neuron added')
    print("centro end" + str(i_iniz))



def plot_prerdecessori(id,n_pred):
    neuron_selected = random.choice(id_neurons_near_to_center[block_plotted[id]], size=(np.min([ id_neurons_near_to_center[block_plotted[id]].__len__(),100])), replace=False)

    spk_list_interval = spk_list[np.logical_and(spk_list[:, 1] > t_initial_analysis, spk_list[:, 1] < t_final_analysis),
                        :]
    spks_times = spk_list_interval[np.in1d(spk_list_interval[:, 0], neuron_selected), 1]
    [hist_value, bins] = np.histogram(spks_times, 1000, [t_initial_analysis, t_final_analysis])

    posizioni = []
    for i in range(len(pos_neuron_list)):
        posizioni.append(f_pos[pos_neuron_list[i]][:])

    posizioni_neuroni = posizioni[0]
    for j in range(1, posizioni.__len__()):
        posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))

    fig3 = go.Figure(data=points, layout=layout)
    fig4 = go.Figure(data=points, layout=layout)

    fig = go.Figure(data=go.Scatter(
        # x=cl_spk[0, np.in1d(col, indici_cl)],
        x=(bins[1:] + bins[:-1]) / 2,
        y=hist_value, # / neuron_selected.__len__(),
        name='voxel_' + str(posizioni_centri[id_center[block_plotted[id]], :]),
        # mode='markers',
        marker=dict(
            # symbol=sim,
            # symbol=142,

            size=2,
            color=0,  # color_discrete_sequence[i],  # set color equal to a variable
            # colorscale='pinkyl', # one of plotly colorscales
            showscale=True
        ),

    ))
    fig_norm = go.Figure(data=go.Scatter(
        # x=cl_spk[0, np.in1d(col, indici_cl)],
        x=(bins[1:] + bins[:-1]) / 2,
        y=hist_value/ neuron_selected.__len__(),
        name='voxel_' + str(posizioni_centri[id_center[block_plotted[id]], :]),
        # mode='markers',
        marker=dict(
            # symbol=sim,
            # symbol=142,

            size=2,
            color=0,  # color_discrete_sequence[i],  # set color equal to a variable
            # colorscale='pinkyl', # one of plotly colorscales
            showscale=True
        ),

    ))

    neuron_to_plt = np.in1d(posizioni_neuroni[:, 0], id_neurons_near_to_center[block_plotted[id]])
    fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                       y=posizioni_neuroni[neuron_to_plt, 2],
                       z=posizioni_neuroni[neuron_to_plt, 3],
                       mode='markers',
                       name='voxel_' + str(posizioni_centri[id_center[block_plotted[id]], :]),
                       text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                       marker=dict(size=2,
                                   # colorscale='pinkyl',
                                   color=color_discrete_sequence[i%color_discrete_sequence.__len__()],
                                   # px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   # showscale=False,
                                   )
                       )

    neuron_to_plt = np.in1d(posizioni_neuroni[:, 0], id_neurons_near_to_center[block_plotted[id]])
    fig4.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                       y=posizioni_neuroni[neuron_to_plt, 2],
                       z=posizioni_neuroni[neuron_to_plt, 3],
                       mode='markers',
                       name='voxel_' + str(posizioni_centri[id_center[block_plotted[id]], :]),
                       text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                       marker=dict(size=2,
                                   # colorscale='pinkyl',
                                   color=color_discrete_sequence[i%color_discrete_sequence.__len__()],
                                   # px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   # showscale=False,
                                   )
                       )

    for i in range(1, n_pred):
        print('generazione', i)
        [st_pyr, st_int, n_st_pyr, n_st_int] = stim_neurons(neuron_selected)
        # st_pyr=st_pyr[:200]
        # st_int = st_int[:200]

        spks_times = spk_list_interval[np.in1d(spk_list_interval[:, 0], st_pyr), 1]
        [hist_value, bins] = np.histogram(spks_times, 1000, [t_initial_analysis, t_final_analysis])
        print("size pyr", st_pyr.__len__())
        print("size int", st_int.__len__())
        spks_times = spk_list_interval[np.in1d(spk_list_interval[:, 0], st_int), 1]
        [hist_value_int, bins] = np.histogram(spks_times, 1000, [t_initial_analysis, t_final_analysis])
        fig.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_value,# / st_pyr.__len__(),
            name='predecessore_' + str(i),  # str(id_center[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i%color_discrete_sequence.__len__()],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

        fig.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_value_int,# / st_int.__len__(),
            name='predecessore_' + str(i) + '_interneurons',  # str(id_center[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i%color_discrete_sequence.__len__()],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

        fig_norm.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_value / st_pyr.__len__(),
            name='predecessore_' + str(i),  # str(id_center[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i%color_discrete_sequence.__len__()],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

        fig_norm.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_value_int / st_int.__len__(),
            name='predecessore_' + str(i) + '_interneurons',  # str(id_center[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i%color_discrete_sequence.__len__()],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

        neuron_to_plt = np.in1d(posizioni_neuroni[:, 0], st_pyr)
        fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                           y=posizioni_neuroni[neuron_to_plt, 2],
                           z=posizioni_neuroni[neuron_to_plt, 3],
                           mode='markers',
                           name='predecessore_' + str(i) + '_pyr',
                           text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                           marker=dict(size=2,
                                       # colorscale='pinkyl',
                                       color=color_discrete_sequence[i%color_discrete_sequence.__len__()],
                                       # px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )

        neuron_to_plt = np.in1d(posizioni_neuroni[:, 0], st_int)
        fig4.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                           y=posizioni_neuroni[neuron_to_plt, 2],
                           z=posizioni_neuroni[neuron_to_plt, 3],
                           mode='markers',
                           name='predecessore_' + str(i) + '_int',
                           text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                           marker=dict(size=2,
                                       # colorscale='pinkyl',
                                       color=color_discrete_sequence[i%color_discrete_sequence.__len__()],
                                       # px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )

        neuron_selected = random.choice(st_pyr, size=(100), replace=False)

    fig.write_html(results_path + "hist_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + "voxel_" + str(
        posizioni_centri[id_center[block_plotted[id]], :]) + "_and_pred.html")
    fig_norm.write_html(results_path + "hist_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + "voxel_" + str(
        posizioni_centri[id_center[block_plotted[id]], :]) + "_and_pred_normalized.html")
    fig3.write_html(results_path + "positions_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + "voxel_" + str(
        posizioni_centri[id_center[block_plotted[id]], :]) + "_and_pred_pyr.html")
    fig4.write_html(results_path + "positions_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + "voxel_" + str(
        posizioni_centri[id_center[block_plotted[id]], :]) + "_and_pred_int.html")




n_sim=1


neu_min_per_blocco=0
neu_min_per_blocco_to_plot=20
sim_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_Giulia_17_10/shuffle_bkg_5hz_5151/"
sim_name="sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b"
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
sim_path= parent_path+"/input_data/sim/"+sim_name+"/"
#sim_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_Giulia_17_10/pruning1_bkg_5hz_5151/"
data_net_path=parent_path+"/input_data/data_net/"
sigla="sl9_"
#sigla="shuffle"
#sigla="pruning1"
# sim_path[1] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
# sigla[1]="ls5"
# sim_path[2] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
# sigla[2]="bkg_5hz"




setting_file = "./configuration.json"
sim_conf = json.load(open('%s' % (setting_file), 'r'))
t_initial_analysis = sim_conf['start_time']
t_final_analysis = sim_conf['end_time']  # 0#5000#
interval_dim = sim_conf['bins_dimension']
n_workers =2
perc_attivi = sim_conf['percentual_of_firing_bins_for_active']
soglia_attivi = perc_attivi * (t_final_analysis - t_initial_analysis) / interval_dim
perc_corr = sim_conf['percentual_of_egual_bins_for_correlation']
soglia_di_correlazione = perc_corr * (t_final_analysis - t_initial_analysis) / interval_dim
n_shift = sim_conf['n_shift']

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


# results_path=sim_path+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
# results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"
# with open(results_sub_path + 'clique_info.pkl', 'rb') as f:
#     [id_neu_cliques, indici_cl, cl_spk, col, Isi_cliques, perc_pyr] = pickle.load(f)
# print("connected components data loaded")


list_of_list_spk=h5py.File(sim_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]
fn=np.unique(spk_list[:,0])




filename_in = data_net_path+"connections_inh.hdf5"
filename_PC = data_net_path+"SP_PC_to_SP_PC.hdf5"
filename_pos=data_net_path+"positions.hdf5"
#with  as f:
f_in=h5py.File(filename_in, "r")
in_connection_list=list(f_in.keys())

f_pyr=h5py.File(filename_PC, "r")
pyr_connection_list=list(f_pyr.keys())

N_pyr=f_pyr[pyr_connection_list[0]][:,0].max()

f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())


distanza_massima=100
bin_x=distanza_massima*2
bin_y=distanza_massima*2
bin_z=distanza_massima*2
min_x=np.min(f_pos[pos_neuron_list[10]][:,1])
min_y=np.min(f_pos[pos_neuron_list[10]][:,2])
min_z=np.min(f_pos[pos_neuron_list[10]][:,3])

max_x=np.max(f_pos[pos_neuron_list[10]][:,1])
max_y=np.max(f_pos[pos_neuron_list[10]][:,2])
max_z=np.max(f_pos[pos_neuron_list[10]][:,3])

results_path=results_path+"area_activity_"+str(bin_x)+"_"+sigla+"/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass


try:
    with open(results_path+'info_firing_per_area_dist_max_'+str(distanza_massima)+'_'+sigla+'.pkl', 'rb') as f:
        [cl_spk_pyr,col_pyr,id_neurons_near_to_center,id_center,posizioni_centri] = pickle.load(f)
except :

    posizioni_centri=[]
    for x in range(np.floor(min_x).astype(int),np.ceil(max_x).astype(int)+bin_x,bin_x):
        for y in range(np.floor(min_y).astype(int), np.ceil(max_y).astype(int) + bin_y, bin_y):
            for z in range(np.floor(min_z).astype(int), np.ceil(max_z).astype(int) + bin_z, bin_z):
                posizioni_centri.append([x,y,z])

    neurons_near_to_center=np.empty(posizioni_centri.__len__(), dtype=object)
    print(posizioni_centri.__len__())
    posizioni_centri=np.array(posizioni_centri[:])

    # for i in range(posizioni_centri.__len__()):
    #     print(i)
    #     distanze_from_pyr=f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i]< distanza_massima
    #     is_comp_dist_min_dist_max = distanze_from_pyr < distanza_massima
    #     is_neuron_near=np.logical_and(is_comp_dist_min_dist_max[:,0],is_comp_dist_min_dist_max[:,1],is_comp_dist_min_dist_max[:,2])
    #     neurons_near_to_center[i]=[]
    #     neurons_near_to_center[i].append(np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3))

    id_center=[]
    id_neurons_near_to_center=[]
    for i in range(posizioni_centri.__len__()):
        print(i)
        #aux=np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3)
        aux = np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:, 1:] - posizioni_centri[i]) < distanza_massima, axis=1) == 3)[0] + 1
        if (aux.__len__()>neu_min_per_blocco):
            id_center.append(i)
            id_neurons_near_to_center.append(aux.tolist())
    #
    # for i in range(posizioni_centri.__len__()):
    #     print(i)
    #     aux=np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3)
    #     neurons_near_to_center[i]=[]
    #     neurons_near_to_center.append(aux[0]+1)

    #interval=1000#np.ceil(posizioni_centri.__len__()/n_workers).astype(int)
    #Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_neuroni_vicini)(t,interval) for t in range(0,posizioni_centri.__len__(),interval))
    #test=26
    print(id_center.__len__())
    [cl_spk_pyr,col_pyr]=save_raster_all2(id_neurons_near_to_center)
    # tot=0
    # for i in range(id_neurons_near_to_center.__len__()):
    #     tot=tot+id_neurons_near_to_center[i].__len__()



    with open(results_path+'info_firing_per_area_dist_max_'+str(distanza_massima)+'_'+sigla+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([cl_spk_pyr,col_pyr,id_neurons_near_to_center,id_center,posizioni_centri] , f)




camera=dict(eye=dict(x=2, y=0, z=0))
points = go.Scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                      y=f_pos[pos_neuron_list[10]][0::10, 2],
                      z=f_pos[pos_neuron_list[10]][0::10, 3],
                      name='network subsampling ',
                      mode='markers',
                      marker=dict(size=1,
                                  color='gray',
                                  #showscale=False,
                                  opacity=0.3),
                      )
layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=10),
                   scene_camera=camera)

fig3 = go.Figure(data=points, layout=layout)
fig4 = go.Figure(data=points, layout=layout)
fig5 = go.Figure(data=points, layout=layout)


min_spk=200
first_clique=True
n_cubi=np.max(col_pyr.astype(int))+1

fr=np.zeros((n_cubi),dtype=float)#np.zeros((np.unique(col_pyr.astype(int)).__len__()),dtype=float)
rad=np.zeros((n_cubi),dtype=float)
period=np.zeros((n_cubi),dtype=float)
hst=np.empty((n_cubi),dtype=object)
block_plotted=[]
for i in np.nditer(np.unique(col_pyr.astype(int))):#range(id_center.__len__()):
    [hist_value, bins] = np.histogram(cl_spk_pyr[0, np.in1d(col_pyr, i)], 1000,[t_initial_analysis,t_final_analysis])
    hst[i]=hist_value
    [f, a, p] = fourier_analysis(hist_value)
    fr[i] = f[np.argmax(a[1:]) + 1]
    rad[i] = p[np.argmax(a[1:]) + 1]
    period[i] = 1 / fr[i]
    print(hist_value.sum()/id_neurons_near_to_center[i].__len__())
    if (np.logical_and((hist_value.sum()/id_neurons_near_to_center[i].__len__())>min_spk,id_neurons_near_to_center[i].__len__()>neu_min_per_blocco_to_plot)):#hist_value.sum()>min_spk:#
        block_plotted.append(i)

        if first_clique:
            first_clique=False
            fig = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:]+bins[:-1])/2,
                y=hist_value,
                name='pyr_cube_'+ str(posizioni_centri[id_center[i],:])+'_'+str(block_plotted.__len__()-1),
                #mode='markers',
                marker=dict(
                    # symbol=sim,
                    #symbol=142,

                    size=2,
                    color=i,#color_discrete_sequence[i],  # set color equal to a variable
                    # colorscale='pinkyl', # one of plotly colorscales
                    showscale=True
                ),

            ))

            fig_norm = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:] + bins[:-1]) / 2,
                y=hist_value/id_neurons_near_to_center[i].__len__(),
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :])+'_'+str(block_plotted.__len__()-1),
                # mode='markers',
                marker=dict(
                    # symbol=sim,
                    # symbol=142,

                    size=2,
                    color=i,  # color_discrete_sequence[i],  # set color equal to a variable
                    # colorscale='pinkyl', # one of plotly colorscales
                    showscale=True
                ),

            ))
        else:
            fig.add_scatter(
                x=(bins[1:] + bins[:-1]) / 2,
                y=hist_value,
                name='pyr_cube_' + str(posizioni_centri[id_center[i],:])+'_'+str(block_plotted.__len__()-1),#str(id_center[i]),
                # mode='markers',
                marker=dict(
                    # symbol=sim,
                    # symbol=142,

                    size=2,
                    color=i,#color_discrete_sequence[i],  # set color equal to a variable
                    # colorscale='pinkyl', # one of plotly colorscales
                    showscale=True
                ),
            )
            fig_norm.add_scatter(
                x=(bins[1:] + bins[:-1]) / 2,
                y=hist_value/id_neurons_near_to_center[i].__len__(),
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :])+'_'+str(block_plotted.__len__()-1),  # str(id_center[i]),
                # mode='markers',
                marker=dict(
                    # symbol=sim,
                    # symbol=142,

                    size=2,
                    color=i,  # color_discrete_sequence[i],  # set color equal to a variable
                    # colorscale='pinkyl', # one of plotly colorscales
                    showscale=True
                ),
            )

        fig3.add_scatter3d(x=[posizioni_centri[id_center[i],0]],
                           y=[posizioni_centri[id_center[i],1]],
                           z=[posizioni_centri[id_center[i],2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),  # str(id_center[i]),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=i/n_cubi,
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )
        print("fr"+str(fr[i]))
        col_max=px.colors.qualitative.Alphabet.__len__()-1
        fig4.add_scatter3d(x=[posizioni_centri[id_center[i], 0]],
                           y=[posizioni_centri[id_center[i], 1]],
                           z=[posizioni_centri[id_center[i], 2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),  # str(id_center[i]),
                           text=str(fr[i]),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[min((fr[i]*10).astype(int),col_max)],
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )
        print("rad" + str(rad[i]))
        fig5.add_scatter3d(x=[posizioni_centri[id_center[i], 0]],
                           y=[posizioni_centri[id_center[i], 1]],
                           z=[posizioni_centri[id_center[i], 2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :])+ '_' + str(block_plotted.__len__() - 1),
                           text=str(fr[i])+' '+str(rad[i]),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[int((rad[i]+np.pi)/(2*np.pi)*col_max)],
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )






        # print("ao")
        # [hist_value_int, bins] = np.histogram(cl_spk_int[0, np.in1d(col_int, indici_cl[i])], 1000,[30000,40000])
        # fig.add_scatter(
        #         x=(bins[1:]+bins[:-1])/2,
        #         y=hist_value_int,
        #         name='int_clique_' + str(indici_cl[i]),
        #         marker=dict(
        #             # symbol=sim,
        #             # symbol=142,
        #
        #             size=2,
        #             color="black",  # set color equal to a variable
        #             # colorscale='pinkyl', # one of plotly colorscales
        #             showscale=True
        #             ),
        #     )
        #
        # ratio = hist_value / hist_value_int
        #
        # fig2.add_scatter(
        #         x=(bins[1:] + bins[:-1]) / 2,
        #         y=ratio,
        #         name='ratio_' + str(indici_cl[i]),
        #         # mode='markers',
        #         marker=dict(
        #             # symbol=sim,
        #             # symbol=142,
        #
        #             size=2,
        #             color=color_discrete_sequence[i],  # set color equal to a variable
        #             # colorscale='pinkyl', # one of plotly colorscales
        #             showscale=True
        #         ),
        #     )
        #
        # dif = hist_value - hist_value_int
        #
        # fig3.add_scatter(
        #     x=(bins[1:] + bins[:-1]) / 2,
        #     y=dif,
        #     name='dif_' + str(indici_cl[i]),
        #     # mode='markers',
        #     marker=dict(
        #         # symbol=sim,
        #         # symbol=142,
        #
        #         size=2,
        #         color=color_discrete_sequence[i],  # set color equal to a variable
        #         # colorscale='pinkyl', # one of plotly colorscales
        #         showscale=True
        #     ),
        # )


fig.write_html(results_path+"hist_firing_" + sigla + "_min_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+".html")
fig_norm.write_html(results_path+"hist_norm_firing_" + sigla + "_min_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+".html")
fig3.write_html(results_path+"position_centers_" + sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+".html")
fig4.write_html(results_path+"position_centers_" + sigla + "_min_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+"_col_freq.html")
fig5.write_html(results_path+"position_centers_" + sigla + "_min_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+"_col_phase.html")

#
# fig2.write_html( "ratio_stim_firing_" + sigla + ".html")
#
# fig3.write_html( "dif_stim_firing_" + sigla + ".html")

piccatura=np.zeros((block_plotted.__len__()),dtype=float)
piccatura2=np.zeros((block_plotted.__len__()),dtype=float)
for i in range(block_plotted.__len__()):
    j=block_plotted[i]
    piccatura[i]=(max(hst[j]) - np.mean(hst[j])) / np.std(hst[j])
    piccatura2[i] = (max(hst[j]) - np.median(hst[j])) / np.std(hst[j])


picchi=np.empty((block_plotted.__len__()),dtype=object)
preminenza=np.empty((block_plotted.__len__()),dtype=object)

picchi_2=np.empty((block_plotted.__len__()),dtype=object)
fase_picchi_2=np.empty((block_plotted.__len__()),dtype=object)
preminenza_2=np.empty((block_plotted.__len__()),dtype=object)

picchi_norm=np.empty((block_plotted.__len__()),dtype=object)
preminenza_norm=np.empty((block_plotted.__len__()),dtype=object)
pr_mean=np.empty((block_plotted.__len__()),dtype=float)
CV_pr=np.empty((block_plotted.__len__()),dtype=float)
preminenza_norm_2=np.empty((block_plotted.__len__()),dtype=object)

CV_pr_2=np.ones((block_plotted.__len__()),dtype=float)*-1
pr_mean_2=np.ones((block_plotted.__len__()),dtype=float)*-1
pr_mean_2_pesato=np.ones((block_plotted.__len__()),dtype=float)*-1
n_picchi=np.zeros((block_plotted.__len__()),dtype=float)
picchi_per_int_find=np.zeros((block_plotted.__len__()),dtype=bool)



sec_di_sim=(t_final_analysis-t_initial_analysis)/1000
from scipy.signal import find_peaks

aux=[8187,1728,7650]
aux=[7987,1728,7650]
aux=[8387,4128,9850]

for i in range(block_plotted.__len__()):
    if((posizioni_centri[id_center[np.array(block_plotted).astype(int)[i]]]==aux).sum()==3):
        print(i)
    peaks, properties = find_peaks(hst[block_plotted[i]], prominence=0)
    preminenza[i] = properties['prominences'][properties['prominences'].argsort()[-int(sec_di_sim / period[block_plotted[i]]):]]#seleziona int(sim_durata/periodo) maggiori preminenze
    picchi[i] = peaks[properties['prominences'].argsort()[-int(sec_di_sim / period[block_plotted[i]]):]]#seleziona int(sim_durata/periodo) picchi con maggior preminenza

    preminenza_2[i]=[]
    picchi_2[i]=[]
    fase_picchi_2[i]=[]

    for j in range(int(sec_di_sim / period[block_plotted[i]])):
        try:
            in_interval=np.logical_and(peaks > j * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]]),peaks < (j + 1) * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]]))
            pos_pr_max=properties['prominences'][in_interval].argmax()
            preminenza_2[i].append(properties['prominences'][in_interval][pos_pr_max]) #seleziona maggiore preminenza del peeriodo i
            picchi_2[i].append(peaks[in_interval][pos_pr_max])
            fase_picchi_2[i].append(peaks[in_interval][pos_pr_max]-j * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]])) #calcolo fase del picco i
            n_picchi[i]=n_picchi[i]+1

        except:
            pass


    #peaks_norm, properties_norm = find_peaks(hst[block_plotted[i]]/id_neurons_near_to_center[block_plotted[i]].__len__(), prominence=0)
    preminenza_norm[i] = preminenza[i]/id_neurons_near_to_center[block_plotted[i]].__len__()
    if (preminenza_2[i].__len__()==int(sec_di_sim / period[block_plotted[i]])):
        picchi_per_int_find[i]=True
        preminenza_norm_2[i]=np.array(preminenza_2[i])/id_neurons_near_to_center[block_plotted[i]].__len__()
        pr_mean_2[i] = preminenza_norm_2[i].mean()
        CV_pr_2[i] = preminenza_norm_2[i].std() / preminenza_norm_2[i].mean()
        pr_mean_2_pesato[i] = pr_mean_2[i] * hst[block_plotted[i]].std() / (hst[block_plotted[i]][1:] - hst[block_plotted[i]][:-1]).std()
    pr_mean[i]=preminenza_norm[i].mean()
    CV_pr[i]=preminenza_norm[i].std()/preminenza_norm[i].mean()

col=np.empty((block_plotted.__len__()),dtype=float)
bp=[]
for i in range(block_plotted.__len__()):
    bp.append(block_plotted[i].tolist())
    col[i]=min(15,int(pr_mean[i]/pr_mean.max() * col_max))


fig7 = go.Figure(data=points, layout=layout)
fig7.add_scatter3d(x=posizioni_centri[np.array(id_center)[bp], 0],
                       y=posizioni_centri[np.array(id_center)[bp], 1],
                       z=posizioni_centri[np.array(id_center)[bp], 2],
                       mode='markers',
                       name='pyr_cube_',
                       text=col,
                       marker=dict(size=5,
                                   colorscale='pinkyl',
                                   color=col,
                                   # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   showscale=True,
                                   )
                       )

fig7.write_html(results_path + "prominence_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + ".html")



corr_fasi=[]#[num_picchi,ind_primo_bloccco,ind_secondo_bloccco,std norm delle diff,mean norm delle diff]
corr_fasi_buone=[]#[num_picchi,ind_primo_bloccco,ind_secondo_bloccco,std norm delle diff,mean norm delle diff]
#corr_fasi_buone2=[]
soglia_ritmicità=0.2 #dato un periodo le distanze tra i picchi devono discostarsi in media al più di soglia_ritmicità*periodo dal periodo
soglia_similarità_picchi=0.5 #i 2 neuroni devono avere un coefficente di variazione della preminenza normalizzata < di soglia_similarità_picchi
soglia_preminenza=0.3 #i 2 neuroni devono avare una preminenza normalizzata >di soglia_preminenza
for j in range(3,10):
    fasi = fase_picchi_2[np.logical_and(n_picchi == j, picchi_per_int_find)]
    bp_aux=np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)]
    for i in range(fasi.__len__()):
        for l in range(i+1,fasi.__len__()):
            corr_fasi.append([j,np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j)),(np.array(fasi[i])-np.array(fasi[l])).mean()/((t_final_analysis-t_initial_analysis)/(10*j))])

            #(dev standard della differenza in fase/ periodo)<soglia,preminenza pesata minore di doglia
            #if np.logical_and(np.logical_and(np.logical_and(j>1,(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j))<0.3),pr_mean_2_pesato[i]>0.4),pr_mean_2_pesato[l]>0.4):
            #    corr_fasi_buone2.append([j,np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j)),(np.array(fasi[i])-np.array(fasi[l])).mean()/((t_final_analysis-t_initial_analysis)/(10*j))])
            preminenza_sopra_soglia1=pr_mean_2_pesato[np.where(np.array(bp)==bp_aux[i])][0]>soglia_preminenza
            preminenza_sopra_soglia2 = pr_mean_2_pesato[np.where(np.array(bp) == bp_aux[l])][0]>soglia_preminenza
            if np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(abs(np.array(fasi[i][1:])-np.array(fasi[i][:-1])).max()/((t_final_analysis-t_initial_analysis)/(10*j))<soglia_ritmicità,abs(np.array(fasi[l][1:])-np.array(fasi[l][:-1])).max()/((t_final_analysis-t_initial_analysis)/(10*j))<soglia_ritmicità),CV_pr_2[i]<soglia_similarità_picchi),CV_pr_2[l]<soglia_similarità_picchi),preminenza_sopra_soglia1),preminenza_sopra_soglia2):
                corr_fasi_buone.append([j, np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i]) - np.array(fasi[l])).std() / ((t_final_analysis - t_initial_analysis) / (10 * j)),(np.array(fasi[i]) - np.array(fasi[l])).mean() / ((t_final_analysis - t_initial_analysis) / (10 * j))])

# i=0
# fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
# pr_mean_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# pr_mean_2[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
#
# pr_mean_2_pesato[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# pr_mean_2_pesato[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
# posizioni_centri[id_center[corr_fasi_buone[i][1]]]
# posizioni_centri[id_center[corr_fasi_buone[i][2]]]
# np.array(fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0])-np.array(fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0])
# #posizioni dei centri di blocchi con firing correlati
# posizioni_centri[id_center[corr_fasi_buone[i][1]]]
# posizioni_centri[id_center[corr_fasi_buone[i][2]]]


line_x=[]
line_y=[]
line_z=[]

for i in range(corr_fasi_buone.__len__()):
    line_x.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][0])
    line_x.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][0])
    line_x.append(None)

    line_y.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][1])
    line_y.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][1])
    line_y.append(None)

    line_z.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][2])
    line_z.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][2])
    line_z.append(None)

fig9 = go.Figure(data=points, layout=layout)
fig9.add_scatter3d(x=line_x,
                   y=line_y,
                   z=line_z,
                   mode='lines',
                   marker=dict(color="green", size=1), opacity=0.7)

for i in range(corr_fasi_buone.__len__()):
    #if corr_fasi_buone[i][-1]>0:
    if np.logical_or(np.logical_and(corr_fasi_buone[i][-1]>0,0.5>corr_fasi_buone[i][-1]),-0.5>corr_fasi_buone[i][-1]):
        fig9.add_cone(x=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][0]],
                            y=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][1]],
                            z=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][2]],
                            u=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][0] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][0]],
                            v=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][1] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][1]],
                            w=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][2] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][2]],
                            sizemode="absolute",
                            sizeref=30,

                            anchor="tip"
                            )
    else:
        fig9.add_cone(x=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0]],
                      y=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1]],
                      z=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2]],
                      u=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0] -posizioni_centri[id_center[corr_fasi_buone[i][1]]][0]],
                      v=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][1]],
                      w=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][2]],
                      sizemode="absolute",
                      sizeref=30,

                      anchor="tip"
                      )

fig9.write_html(results_path + "prominence_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + ".html")


#color_discrete_sequence = ["black","orange", "red", "green", "blue", "pink","yellow"]
color_discrete_sequence=px.colors.qualitative.Alphabet
id=1
n_predecessori=2
plot_prerdecessori(id,n_predecessori)
#neuron_selected=id_neurons_near_to_center[block_plotted[id]]
#
# from dash import Dash, dcc, html, Input, Output
# from dash.dependencies import State
# import plotly.express as px
#
# app = Dash(__name__)
#
#
#
#
# app.layout = html.Div([
#     html.H4('concentration'),
#     dcc.Graph(id="graph"),
#     html.P("variabilità preminenza dei picchi:"),
#     dcc.RangeSlider(
#         id='range-slider',
#         min=0, max=CV_pr_2.max(), step=0.1,
#         marks={0: '0', CV_pr_2.max(): str(CV_pr_2.max())},
#         value=[0, CV_pr_2.max()]
#     ),
#     html.P("preminenza media dei pricchi:"),
#     dcc.RangeSlider(
#         id='range-slider2',
#         min=0, max=pr_mean_2.max(), step=0.1,
#         marks={0: '0', pr_mean_2.max(): str(pr_mean_2.max())},
#         value=[0, pr_mean_2.max()]
#     ),
# ])
#
# @app.callback(
#     Output("graph", "figure"),
#     Input("range-slider", "value"),
#     Input("range-slider2", "value"),
# )
# def update_bar_chart(slider_range,slider_range2):
#     #df = px.data.iris() # replace with your own data source
#     global f_pos,pos_neuron_list,neurons_selected,fig
#     low_CV_pr, high_CV_pr = slider_range
#     low_pr_mean,high_pr_mean = slider_range2
#
#
#     fig = go.Figure(data=points, layout=layout)
#     # fig = px.scatter_3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
#     #                     y=f_pos[pos_neuron_list[10]][0::10, 2],
#     #                     z=f_pos[pos_neuron_list[10]][0::10, 3],
#     #                     size=np.ones(f_pos[pos_neuron_list[10]][0::10, 3].__len__()) * 0.3,
#     #                     opacity=0.5, size_max=1)
#
#
#     mask = np.logical_and(np.logical_and(CV_pr_2 > low_CV_pr, CV_pr_2 < high_CV_pr),np.logical_and(pr_mean_2 > low_pr_mean, pr_mean_2 < high_pr_mean))
#     fig.add_scatter3d(x=posizioni_centri[np.array(id_center)[np.array(bp)[mask]], 0],
#                        y=posizioni_centri[np.array(id_center)[np.array(bp)[mask]], 1],
#                        z=posizioni_centri[np.array(id_center)[np.array(bp)[mask]], 2],
#                        mode='markers',
#                        name='pyr_cube_',
#                        text=pr_mean_2[mask],
#                        marker=dict(size=5,
#                                    colorscale='pinkyl',
#                                    color=pr_mean_2[mask],
#                                    # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
#                                    # colorscale='Plotly3',
#                                    showscale=True,
#                                    )
#                        )
#
#     # if type_to_plot == "pyr":
#     #     fig.write_html(path_res + "concentration_map_pyr.html")
#     # else:
#     #     fig.write_html(path_res + "concentration_map_int.html")
#
#     return fig
#
#
# app.run_server(port=8022)
#
