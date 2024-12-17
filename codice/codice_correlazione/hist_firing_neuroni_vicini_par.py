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








n_sim=1



work_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
sigla="sl9_"
# work_path[1] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
# sigla[1]="ls5"
# work_path[2] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
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

results_path=work_path+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"
with open(results_sub_path + 'clique_info.pkl', 'rb') as f:
    [id_neu_cliques, indici_cl, cl_spk, col, Isi_cliques, perc_pyr] = pickle.load(f)
print("connected components data loaded")


list_of_list_spk=h5py.File(work_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]
fn=np.unique(spk_list[:,0])




filename_in = "connections_inh.hdf5"
filename_PC = "SP_PC_to_SP_PC.hdf5"
filename_pos="positions.hdf5"
#with  as f:
f_in=h5py.File(filename_in, "r")
in_connection_list=list(f_in.keys())

f_pyr=h5py.File(filename_PC, "r")
pyr_connection_list=list(f_pyr.keys())

N_pyr=f_pyr[pyr_connection_list[0]][:,0].max()

f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())


distanza_massima=50
bin_x=distanza_massima*2
bin_y=distanza_massima*2
bin_z=distanza_massima*2
min_x=np.min(f_pos[pos_neuron_list[10]][:,1])
min_y=np.min(f_pos[pos_neuron_list[10]][:,2])
min_z=np.min(f_pos[pos_neuron_list[10]][:,3])

max_x=np.max(f_pos[pos_neuron_list[10]][:,1])
max_y=np.max(f_pos[pos_neuron_list[10]][:,2])
max_z=np.max(f_pos[pos_neuron_list[10]][:,3])

results_path="./area_activity_"+str(bin_x)+"_"+sigla+"/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass


try:
    with open(results_path+'1info_firing_per_area_dist_max_'+str(distanza_massima)+'_'+sigla+'.pkl', 'rb') as f:
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
        if (aux.__len__()>0):
            id_center.append(i)
            id_neurons_near_to_center.append(aux.tolist())
    #
    # for i in range(posizioni_centri.__len__()):
    #     print(i)
    #     aux=np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3)
    #     neurons_near_to_center[i]=[]
    #     neurons_near_to_center.append(aux[0]+1)

    interval=10#np.ceil(posizioni_centri.__len__()/n_workers).astype(int)
    #Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_neuroni_vicini)(t,interval) for t in range(0,posizioni_centri.__len__(),interval))
    #test=26
    n_workers=4
    Ltt = Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(save_raster_all2)(id_neurons_near_to_center[t:t+interval]) for t in range(0, posizioni_centri[:85].__len__()))
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

min_spk=50
first_clique=True
n_cubi=np.max(col_pyr.astype(int))+1

fr=np.zeros((n_cubi),dtype=float)#np.zeros((np.unique(col_pyr.astype(int)).__len__()),dtype=float)
rad=np.zeros((n_cubi),dtype=float)
period=np.zeros((n_cubi),dtype=float)
block_plotted=[]
for i in np.nditer(np.unique(col_pyr.astype(int))):#range(id_center.__len__()):
    [hist_value, bins] = np.histogram(cl_spk_pyr[0, np.in1d(col_pyr, i)], 1000,[30000,40000])
    [f, a, p] = fourier_analysis(hist_value)
    fr[i] = f[np.argmax(a[1:]) + 1]
    rad[i] = p[np.argmax(a[1:]) + 1]
    period[i] = 1 / fr[i]
    print(hist_value.sum()/id_neurons_near_to_center[i].__len__())
    if (hist_value.sum()/id_neurons_near_to_center[i].__len__())>min_spk:#
        block_plotted.append(i)
        if first_clique:
            first_clique=False
            fig = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:]+bins[:-1])/2,
                y=hist_value,
                name='pyr_cube_'+ str(posizioni_centri[id_center[i],:]),
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
        else:
            fig.add_scatter(
                x=(bins[1:] + bins[:-1]) / 2,
                y=hist_value,
                name='pyr_cube_' + str(posizioni_centri[id_center[i],:]),#str(id_center[i]),
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

        fig3.add_scatter3d(x=[posizioni_centri[id_center[i],0]],
                           y=[posizioni_centri[id_center[i],1]],
                           z=[posizioni_centri[id_center[i],2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i],:]),
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
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]),
                           text=str(fr[i]),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[min((fr[i]*10).astype(int),col_max)],
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


fig.write_html(results_path+"hist_firing_" + sigla + "_min_"+str(min_spk)+".html")
fig3.write_html(results_path+"position_centers_" + sigla + "_min_"+str(min_spk)+".html")
fig4.write_html(results_path+"position_centers_" + sigla + "_min_"+str(min_spk)+"_col_freq.html")
#
# fig2.write_html( "ratio_stim_firing_" + sigla + ".html")
#
# fig3.write_html( "dif_stim_firing_" + sigla + ".html")