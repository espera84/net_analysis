import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import h5py
from joblib import Parallel, delayed
import json
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

    return freq, amplitude, phase,dft


def calcola_spk_id(j,list_id_neuroni):
    global t_initial_analysis, t_final_analysis
    i = 0
    id_neuroni = list_id_neuroni[j]
    first_neuron = True
    print("lista", j)
    print("lung lista", id_neuroni.__len__())
    # spk_id=spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0],1]
    spk_id = np.vstack((spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0], 1],
                        spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0], 0]))
    spk_id_interval = spk_id[:, np.logical_and(spk_id[0, :] > t_initial_analysis, spk_id[0, :] < t_final_analysis)]
    #for i in range(list_id_neuroni[j].__len__()):
    #    spk_id_interval[1, np.where(spk_id_interval[1, :] == list_id_neuroni[j][i])] = k + i
    #k = k + i + 1
    color=np.ones(spk_id_interval[0,:].__len__())*j
    return spk_id_interval,color
def select_spk_for_voxel(list_id_neuroni):

    global t_initial_analysis,t_final_analysis

    n_workers=10
    aux=Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_spk_id)(t,list_id_neuroni) for t in range(list_id_neuroni.__len__()))
    #[spk, color]=aux

    for i in range(len(aux)):
        if i == 0:
            #cl_spk = np.vstack((np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__()))), np.ones(spk[i].__len__()) * i))
            cl_spk = aux[i][0]
            color =aux[i][1]
            #cl_spk = np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))
        else:
            # print(np.ones(spk[i].__len__()))
            #cl_spk = np.concatenate((cl_spk, np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))), axis=1)
            cl_spk = np.concatenate((cl_spk,aux[i][0]) , axis=1)
            color = np.concatenate((color,aux[i][1]) , axis=0)



    return cl_spk,color


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





current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
parent_path = os.path.dirname(parent_path)

#sim_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_Giulia_17_10/pruning1_bkg_5hz_5151/"
data_net_path=parent_path+"/input_data/data_net/"
sigla="sl9_5-1-5-1"#"sl9_9-1-9-1"
#sigla="shuffle"
#sigla="pruning1"
# sim_path[1] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
# sigla[1]="ls5"
# sim_path[2] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
# sigla[2]="bkg_5hz"




setting_file = "./h_f_configuration.json"
sim_conf = json.load(open('%s' % (setting_file), 'r'))
t_initial_analysis = sim_conf['start_time']
t_final_analysis = sim_conf['end_time']  # 0#5000#
modality=sim_conf['neuron_type_to_analize']#'all'#'pyr'#'int'#
neu_min_per_blocco=0
neu_min_per_blocco_to_plot=sim_conf['minimum_number_of_neurons_for_voxel_to_plot']#100
sim_name=sim_conf["folder_simulation_name"]#"sl9_9-1-9-1"
distanza_massima=int(sim_conf['side_length_of_the_voxel']/2)
min_spk=sim_conf["minimum_mean_number_of_spikes_for_neuron_in_voxel_to_plot"]
bin_size = sim_conf['bins_dimension']

sim_path= parent_path+"/input_data/sim/"+sim_name+"/"
network_results_path=parent_path+"/results/network/"
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



bin_x=int(sim_conf["distance_between_voxel_centers"])
bin_y=int(sim_conf["distance_between_voxel_centers"])
bin_z=int(sim_conf["distance_between_voxel_centers"])
min_x=np.inf
min_y=np.inf
min_z=np.inf
max_x=-np.inf
max_y=-np.inf
max_z=-np.inf
for j in range(pos_neuron_list.__len__()):
    min_x=min(min_x,np.min(f_pos[pos_neuron_list[j]][:,1]))
    min_y=min(min_y,np.min(f_pos[pos_neuron_list[j]][:,2]))
    min_z=min(min_z,np.min(f_pos[pos_neuron_list[j]][:,3]))

    max_x=max(max_x,np.max(f_pos[pos_neuron_list[j]][:,1]))
    max_y=max(max_y,np.max(f_pos[pos_neuron_list[j]][:,2]))
    max_z=max(max_z,np.max(f_pos[pos_neuron_list[j]][:,3]))

results_path=results_path+"area_activity_dist_cent_"+str(bin_x)+"_voxel_sl_"+str(distanza_massima*2)+"_"+sigla+"/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass


try:
    with open(network_results_path+'info_neurons_distribution_in_voxels_area_dist_max_'+str(distanza_massima)+'_dist_cent_'+str(bin_x)+'.pkl', 'rb') as f:
        [id_neurons_near_to_center,id_neurons_near_to_center_pyr,id_neurons_near_to_center_int,id_center,posizioni_centri] = pickle.load(f)
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
    id_neurons_near_to_center_pyr = []
    id_neurons_near_to_center_int = []
    for i in range(posizioni_centri.__len__()):
        print(i)
        #aux=np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3)
        aux=[]
        aux_int=[]
        aux_pyr =[]
        for j in range(pos_neuron_list.__len__()):
            if j==10:
                aux_pyr=f_pos[pos_neuron_list[j]][np.where(np.sum(np.abs(f_pos[pos_neuron_list[j]][:, 1:] - posizioni_centri[i]) < distanza_massima, axis=1) == 3)[0],0].astype(int).tolist()
            else:
                aux_int=aux_int+f_pos[pos_neuron_list[j]][np.where(np.sum(np.abs(f_pos[pos_neuron_list[j]][:, 1:] - posizioni_centri[i]) < distanza_massima, axis=1) == 3)[0],0].astype(int).tolist()
            #aux = aux+f_pos[pos_neuron_list[j]][np.where(np.sum(np.abs(f_pos[pos_neuron_list[j]][:, 1:] - posizioni_centri[i]) < distanza_massima, axis=1) == 3)[0],0].astype(int).tolist()
        aux=aux_pyr+aux_int
        #print(f_pos[pos_neuron_list[10]][np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:, 1:] - posizioni_centri[i]) < distanza_massima, axis=1) == 3)[0],0].astype(int).tolist())
        if (aux.__len__()>neu_min_per_blocco):
            id_center.append(i)
            id_neurons_near_to_center.append(aux)
            id_neurons_near_to_center_pyr.append(aux_pyr)
            id_neurons_near_to_center_int.append(aux_int)
    # for i in range(posizioni_centri.__len__()):
    #     print(i)
    #     aux=np.where(np.sum(np.abs(f_pos[pos_neuron_list[10]][:,1:]-posizioni_centri[i])< distanza_massima,axis=1)==3)
    #     neurons_near_to_center[i]=[]
    #     neurons_near_to_center.append(aux[0]+1)

    print(id_center.__len__())
    with open(network_results_path+'info_neurons_distribution_in_voxels_area_dist_max_'+str(distanza_massima)+'_dist_cent_'+str(bin_x)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([id_neurons_near_to_center,id_neurons_near_to_center_pyr,id_neurons_near_to_center_int,id_center,posizioni_centri] , f)
try:
    with open(results_path + 'info_firing_per_area_dist_max_' + str(distanza_massima) +'_dist_cent_'+str(bin_x)+'_sigla_'+sigla+ '.pkl', 'rb') as f:
        [voxel_spk_all,id_voxel_for_spk_all,voxel_spk_pyr, id_voxel_for_spk_pyr,voxel_spk_int, id_voxel_for_spk_int] = pickle.load(f)
except:
    # voxel_spk_all sono gli spikes voxel_spk[0] con gli id dei neuroni associati voxel_spk[1] nell'intervallo selezionato dei neuroni presenti in id_neurons_near_to_center, select_spk_for_voxel contiene gli id dei voxel in cui si verificano gli spikes presenti in voxel_spk_all
    [voxel_spk_all,id_voxel_for_spk_all]=select_spk_for_voxel(id_neurons_near_to_center)
    [voxel_spk_pyr, id_voxel_for_spk_pyr] = select_spk_for_voxel(id_neurons_near_to_center_pyr)
    [voxel_spk_int, id_voxel_for_spk_int] = select_spk_for_voxel(id_neurons_near_to_center_int)
    # tot=0
    # for i in range(id_neurons_near_to_center.__len__()):
    #     tot=tot+id_neurons_near_to_center[i].__len__()



    with open(results_path+'info_firing_per_area_dist_max_'+str(distanza_massima)+'_dist_cent_'+str(bin_x)+'_sigla_'+sigla+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([voxel_spk_all,id_voxel_for_spk_all,voxel_spk_pyr, id_voxel_for_spk_pyr,voxel_spk_int, id_voxel_for_spk_int] , f)

if modality=='all':
    id_voxel_for_spk=id_voxel_for_spk_all
    voxel_spk=voxel_spk_all
if modality=='pyr':
    id_voxel_for_spk=id_voxel_for_spk_pyr
    voxel_spk=voxel_spk_pyr
if modality=='int':
    id_voxel_for_spk=id_voxel_for_spk_int
    voxel_spk=voxel_spk_int

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
fig6 = go.Figure(data=points, layout=layout)
fig7 = go.Figure(data=points, layout=layout)



first_clique=True
n_cubi=np.max(id_voxel_for_spk.astype(int))+1

fr=np.zeros((n_cubi),dtype=float)
rad=np.zeros((n_cubi),dtype=float)
rad_filtered=np.zeros((n_cubi),dtype=float)
period=np.zeros((n_cubi),dtype=float)
hst=np.empty((n_cubi),dtype=object)
filtered_sig=np.empty((n_cubi),dtype=object)
perc_amp_max=np.empty((n_cubi),dtype=object)
#perc_amp_max_tot=np.empty((n_cubi),dtype=object)
block_plotted=[]

buond_down_filter=sim_conf["freq_buond_down"]
buond_up_filter=sim_conf["freq_buond_up"]
threshold_perc_amp=sim_conf["threshold_perc_amp"]
pos_c_x=[]
pos_c_y=[]
pos_c_z=[]
color_phase=[]
n_bin=int((t_final_analysis-t_initial_analysis)/bin_size)
for i in np.nditer(np.unique(id_voxel_for_spk.astype(int))):#range(id_center.__len__()):
    [hist_value, bins] = np.histogram(voxel_spk[0, np.in1d(id_voxel_for_spk, i)], n_bin,[t_initial_analysis,t_final_analysis])
    hst[i]=hist_value
    [f, a, p,dft] = fourier_analysis(hist_value)
    fr[i] = f[np.argmax(a[1:]) + 1]
    rad[i] = p[np.argmax(a[1:]) + 1]
    perc_amp_max[i]=a[np.argmax(a[1:])+1]/a[1:].sum()
    #perc_amp_max_tot[i]=a[np.argmax(a[1:])]/a[1:].sum()
    period[i] = 1 / fr[i]
    dft[f > buond_up_filter] = 0
    dft[f < buond_down_filter] = 0
    filtered_sig[i]=np.fft.irfft(dft)
    rad_filtered[i]=p[np.logical_and(f < buond_up_filter, f > buond_down_filter)][0]
    print(hist_value.sum()/id_neurons_near_to_center[i].__len__())
    if (np.logical_and((hist_value.sum()/id_neurons_near_to_center[i].__len__())>min_spk,id_neurons_near_to_center[i].__len__()>neu_min_per_blocco_to_plot)):#hist_value.sum()>min_spk:#
        block_plotted.append(i)
        color_phase .append(int((rad[i] + np.pi) / (2 * np.pi) * 36))
        pos_c_x.append(posizioni_centri[id_center[i], 0])
        pos_c_y.append(posizioni_centri[id_center[i], 1])
        pos_c_z.append(posizioni_centri[id_center[i], 2])
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

            fig_up = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:] + bins[:-1]) / 2,
                y=hist_value,
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),
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


            fig_filtered = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:] + bins[:-1]) / 2,
                y=filtered_sig[i],
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),
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

            fig_filtered_norm = go.Figure(data=go.Scatter(
                # x=cl_spk[0, np.in1d(col, indici_cl)],
                x=(bins[1:] + bins[:-1]) / 2,
                y=filtered_sig[i]/filtered_sig[i].max(),
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),
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
            fig_filtered.add_scatter(
                x=(bins[1:] + bins[:-1]) / 2,
                y=filtered_sig[i],
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),
                # str(id_center[i]),
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
            fig_filtered_norm.add_scatter(
                x=(bins[1:] + bins[:-1]) / 2,
                y=filtered_sig[i]/filtered_sig[i].max(),
                name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),
                # str(id_center[i]),
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

        col_max=px.colors.qualitative.Alphabet.__len__()-1
        fig4.add_scatter3d(x=[posizioni_centri[id_center[i], 0]],
                           y=[posizioni_centri[id_center[i], 1]],
                           z=[posizioni_centri[id_center[i], 2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(block_plotted.__len__() - 1),  # str(id_center[i]),
                           text=str(fr[i])+' '+str(rad[i])+' '+ str(block_plotted.__len__() - 1),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[min((fr[i]*10).astype(int),col_max)],
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )
        if perc_amp_max[i]>threshold_perc_amp:
            fig6.add_scatter3d(x=[posizioni_centri[id_center[i], 0]],
                           y=[posizioni_centri[id_center[i], 1]],
                           z=[posizioni_centri[id_center[i], 2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(
                               block_plotted.__len__() - 1),  # str(id_center[i]),
                           text=str(fr[i]) + ' ' + str(rad[i]) + ' ' + str(block_plotted.__len__() - 1),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[min((fr[i] * 10).astype(int), col_max)],
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )
        fig7.add_scatter3d(x=[posizioni_centri[id_center[i], 0]],
                           y=[posizioni_centri[id_center[i], 1]],
                           z=[posizioni_centri[id_center[i], 2]],
                           mode='markers',
                           name='pyr_cube_' + str(posizioni_centri[id_center[i], :]) + '_' + str(
                               block_plotted.__len__() - 1),  # str(id_center[i]),
                           text=str(fr[i]) + ' ' + str(rad[i]) + ' ' + str(block_plotted.__len__() - 1),
                           marker=dict(size=5,
                                       # colorscale='pinkyl',
                                       color=px.colors.qualitative.Alphabet[min((perc_amp_max[i] * 100).astype(int), col_max)],
                                       # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       # colorscale='Plotly3',
                                       # showscale=False,
                                       )
                           )

try:
    fig.write_html(results_path+"hist_firing_" + sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+modality+".html")
    fig_filtered.write_html(results_path + "hist_firing_filtered_" + sigla + "_min_neu_spk_" + str(min_spk) + "_min_neu_Xb_" + str(neu_min_per_blocco_to_plot) + modality +"_"+str(buond_down_filter)+"_"+str(buond_up_filter)+ "_HZ.html")
    fig_norm.write_html(results_path+"hist_norm_firing_" + sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+modality+".html")
    fig_filtered_norm.write_html(results_path + "hist_norm_firing_filtered_" + sigla + "_min_neu_spk_" + str(min_spk) + "_min_neu_Xb_" + str(neu_min_per_blocco_to_plot) + modality +"_"+str(buond_down_filter)+"_"+str(buond_up_filter)+ "_HZ.html")
except:
    pass
fig4.write_html(results_path+"position_centers_" + sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+modality+"_col_freq.html")
fig6.write_html(results_path+"position_centers_th_perc_amp_"+str(threshold_perc_amp)+"_"+ sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+modality+"_col_freq.html")
fig7.write_html(results_path+"position_centers_th_perc_amp_"+str(threshold_perc_amp)+"_"+ sigla + "_min_neu_spk_"+str(min_spk)+"_min_neu_Xb_"+str(neu_min_per_blocco_to_plot)+modality+".html")


col=np.empty((block_plotted.__len__()),dtype=float)
colphase=np.empty((block_plotted.__len__()),dtype=float)
colphase_filtered=np.empty((block_plotted.__len__()),dtype=float)
n_blk_over_th=np.sum(perc_amp_max[block_plotted]>threshold_perc_amp)
n_blk_filtered_over_th=np.sum(perc_amp_max[block_plotted]>threshold_perc_amp)
colphase_over_th=np.empty((n_blk_over_th),dtype=float)
colphase_filtered_over_th=np.empty((n_blk_filtered_over_th),dtype=float)
vicini=np.empty((block_plotted.__len__()),dtype=object)


bp=[]
bp_over_th=[]
j=0
for i in range(block_plotted.__len__()):
    bp.append(block_plotted[i].tolist())
    colphase[i] = (rad[block_plotted[i]] + np.pi) #/ (2 * np.pi)#int((rad[block_plotted[i]] + np.pi) / (2 * np.pi) * 36)
    colphase_filtered[i] = (rad_filtered[block_plotted[i]] + np.pi)  # / (2 * np.pi)#int((rad[block_plotted[i]] + np.pi) / (2 * np.pi) * 36)
    if perc_amp_max[block_plotted[i]]>threshold_perc_amp:
        colphase_over_th[j]=(rad[block_plotted[i]] + np.pi)
        bp_over_th.append(block_plotted[i].tolist())
        j=j+1
for i in range(block_plotted.__len__()):
    vicini[i]=np.where(np.sum(np.abs(posizioni_centri[np.array(id_center)[bp][i],:]-posizioni_centri[np.array(id_center)[bp],:])<=[bin_x,bin_y,bin_z],1)==3)[0][np.where(np.sum(np.abs(posizioni_centri[np.array(id_center)[bp][i],:]-posizioni_centri[np.array(id_center)[bp],:])<=[bin_x,bin_y,bin_z],1)==3)[0]!=i]
    #vicini[i]=np.where(np.sum(np.abs(posizioni_centri[np.array(id_center)[bp][i],:]-posizioni_centri[np.array(id_center)[bp],:])<=[100,100,100],1)==3)





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
        neu_min_per_blocco_to_plot) + modality+".html")

fig8 = go.Figure(data=points, layout=layout)
fig8.add_scatter3d(x=posizioni_centri[np.array(id_center)[bp], 0],
                       y=posizioni_centri[np.array(id_center)[bp], 1],
                       z=posizioni_centri[np.array(id_center)[bp], 2],
                       mode='markers',
                       name='pyr_cube_',
                       text=colphase,
                       marker=dict(size=5,
                                   colorscale='Phase',
                                   color=colphase,
                                   # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   showscale=True,
                                   )
                       )

fig8.write_html(results_path + "phase_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + modality+".html")

fig8 = go.Figure(data=points, layout=layout)
fig8.add_scatter3d(x=posizioni_centri[np.array(id_center)[bp], 0],
                       y=posizioni_centri[np.array(id_center)[bp], 1],
                       z=posizioni_centri[np.array(id_center)[bp], 2],
                       mode='markers',
                       name='pyr_cube_',
                       text=colphase_filtered,
                       marker=dict(size=5,
                                   colorscale='Phase',
                                   color=colphase_filtered,
                                   # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   showscale=True,
                                   )
                       )

fig8.write_html(results_path + "phase_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + modality+"_"+str(buond_down_filter)+"_"+str(buond_up_filter)+ "_HZ.html")

fig8 = go.Figure(data=points, layout=layout)
fig8.add_scatter3d(x=posizioni_centri[np.array(id_center)[bp_over_th], 0],
                       y=posizioni_centri[np.array(id_center)[bp_over_th], 1],
                       z=posizioni_centri[np.array(id_center)[bp_over_th], 2],
                       mode='markers',
                       name='pyr_cube_',
                       text=colphase_over_th,
                       marker=dict(size=5,
                                   colorscale='Phase',
                                   color=colphase_over_th,
                                   # color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   # colorscale='Plotly3',
                                   showscale=True,
                                   )
                       )

fig8.write_html(results_path + "phase_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
        neu_min_per_blocco_to_plot) + modality+"_over_th.html")


vicino_max_phase_var=np.empty((block_plotted.__len__()),dtype=int)

line_x=[]
line_y=[]
line_z=[]
for i in range(block_plotted.__len__()):
    if vicini[i].__len__()>0:
        aux=colphase_filtered[vicini[i]]-colphase_filtered[i]
        aux[aux > np.pi] = aux[aux > np.pi] - (2 * np.pi)
        aux[aux < -np.pi] = aux[aux < -np.pi] + (2 * np.pi)
        for j in range(vicini[i].__len__()):
            aux[j]=aux[j]/np.linalg.norm(np.abs(posizioni_centri[np.array(id_center)[bp[i]]]-posizioni_centri[np.array(id_center)[bp[vicini[i][j]]]])/bin_x)
        vicino_max_phase_var[i]=aux.argmax()

        line_x.append(posizioni_centri[np.array(id_center)[bp[i]]][0])
        line_x.append(posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][0])
        line_x.append(None)

        line_y.append(posizioni_centri[np.array(id_center)[bp[i]]][1])
        line_y.append(posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][1])
        line_y.append(None)

        line_z.append(posizioni_centri[np.array(id_center)[bp[i]]][2])
        line_z.append(posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][2])
        line_z.append(None)

fig9 = go.Figure(data=points, layout=layout)
fig9.add_scatter3d(x=line_x,
                   y=line_y,
                   z=line_z,
                   mode='lines',
                   marker=dict(color="green", size=1), opacity=0.7)

for i in range(block_plotted.__len__()):
    if vicini[i].__len__():
        fig9.add_cone(x=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][0]],
                  y=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][1]],
                  z=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][2]],
                  u=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][0] - posizioni_centri[np.array(id_center)[bp[i]]][0]],
                  v=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][1] - posizioni_centri[np.array(id_center)[bp[i]]][1]],
                  w=[posizioni_centri[np.array(id_center)[bp[vicini[i][vicino_max_phase_var[i]]]]][2] - posizioni_centri[np.array(id_center)[bp[i]]][2]],
                  sizemode="absolute",
                  sizeref=30,
                  anchor="tip"
                  )
fig9.write_html(results_path + "phase_path_" + sigla + "_filtered_between_" + str(buond_down_filter) + "_and_" + str(buond_up_filter) + ".html")
#     else:
#         fig9.add_cone(x=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0]],
#                       y=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1]],
#                       z=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2]],
#                       u=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0] -posizioni_centri[id_center[corr_fasi_buone[i][1]]][0]],
#                       v=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][1]],
#                       w=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][2]],
#                       sizemode="absolute",
#                       sizeref=30,
#
#                       anchor="tip"
#                       )
#
# fig9.write_html(results_path + "prominence_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
#         neu_min_per_blocco_to_plot) + ".html")

#
#
#
#
# piccatura=np.zeros((block_plotted.__len__()),dtype=float)
# piccatura2=np.zeros((block_plotted.__len__()),dtype=float)
# for i in range(block_plotted.__len__()):
#     j=block_plotted[i]
#     piccatura[i]=(max(hst[j]) - np.mean(hst[j])) / np.std(hst[j])
#     piccatura2[i] = (max(hst[j]) - np.median(hst[j])) / np.std(hst[j])
#
#
# picchi=np.empty((block_plotted.__len__()),dtype=object)
# preminenza=np.empty((block_plotted.__len__()),dtype=object)
#
# picchi_2=np.empty((block_plotted.__len__()),dtype=object)
# fase_picchi_2=np.empty((block_plotted.__len__()),dtype=object)
# preminenza_2=np.empty((block_plotted.__len__()),dtype=object)
#
# picchi_norm=np.empty((block_plotted.__len__()),dtype=object)
# preminenza_norm=np.empty((block_plotted.__len__()),dtype=object)
# pr_mean=np.empty((block_plotted.__len__()),dtype=float)
# CV_pr=np.empty((block_plotted.__len__()),dtype=float)
# preminenza_norm_2=np.empty((block_plotted.__len__()),dtype=object)
#
# CV_pr_2=np.ones((block_plotted.__len__()),dtype=float)*-1
# pr_mean_2=np.ones((block_plotted.__len__()),dtype=float)*-1
# pr_mean_2_pesato=np.ones((block_plotted.__len__()),dtype=float)*-1
# n_picchi=np.zeros((block_plotted.__len__()),dtype=float)
# picchi_per_int_find=np.zeros((block_plotted.__len__()),dtype=bool)
#
#
#
# sec_di_sim=(t_final_analysis-t_initial_analysis)/1000
# from scipy.signal import find_peaks
#
# #aux=[8187,1728,7650]
# #aux=[7987,1728,7650]
# #aux=[8387,4128,9850]
#
# for i in range(block_plotted.__len__()):
#     #if((posizioni_centri[id_center[np.array(block_plotted).astype(int)[i]]]==aux).sum()==3):
#     #    print(i)
#     peaks, properties = find_peaks(hst[block_plotted[i]], prominence=0)
#     preminenza[i] = properties['prominences'][properties['prominences'].argsort()[-int(sec_di_sim / period[block_plotted[i]]):]]#seleziona int(sim_durata/periodo) maggiori preminenze
#     picchi[i] = peaks[properties['prominences'].argsort()[-int(sec_di_sim / period[block_plotted[i]]):]]#seleziona int(sim_durata/periodo) picchi con maggior preminenza
#
#     preminenza_2[i]=[]
#     picchi_2[i]=[]
#     fase_picchi_2[i]=[]
#
#     for j in range(int(sec_di_sim / period[block_plotted[i]])):
#         try:
#             in_interval=np.logical_and(peaks > j * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]]),peaks < (j + 1) * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]]))
#             pos_pr_max=properties['prominences'][in_interval].argmax()
#             preminenza_2[i].append(properties['prominences'][in_interval][pos_pr_max]) #seleziona maggiore preminenza del peeriodo i
#             picchi_2[i].append(peaks[in_interval][pos_pr_max])
#             fase_picchi_2[i].append(peaks[in_interval][pos_pr_max]-j * hst[block_plotted[i]].__len__() / int(sec_di_sim / period[block_plotted[i]])) #calcolo fase del picco i
#             n_picchi[i]=n_picchi[i]+1
#
#         except:
#             pass
#
#
#     #peaks_norm, properties_norm = find_peaks(hst[block_plotted[i]]/id_neurons_near_to_center[block_plotted[i]].__len__(), prominence=0)
#     preminenza_norm[i] = preminenza[i]/id_neurons_near_to_center[block_plotted[i]].__len__()
#     if (preminenza_2[i].__len__()==int(sec_di_sim / period[block_plotted[i]])):
#         picchi_per_int_find[i]=True
#         preminenza_norm_2[i]=np.array(preminenza_2[i])/id_neurons_near_to_center[block_plotted[i]].__len__()
#         pr_mean_2[i] = preminenza_norm_2[i].mean()
#         CV_pr_2[i] = preminenza_norm_2[i].std() / preminenza_norm_2[i].mean()
#         pr_mean_2_pesato[i] = pr_mean_2[i] * hst[block_plotted[i]].std() / (hst[block_plotted[i]][1:] - hst[block_plotted[i]][:-1]).std()
#     pr_mean[i]=preminenza_norm[i].mean()
#     CV_pr[i]=preminenza_norm[i].std()/preminenza_norm[i].mean()
#
#
# corr_fasi=[]#[num_picchi,ind_primo_bloccco,ind_secondo_bloccco,std norm delle diff,mean norm delle diff]
# corr_fasi_buone=[]#[num_picchi,ind_primo_bloccco,ind_secondo_bloccco,std norm delle diff,mean norm delle diff]
# #corr_fasi_buone2=[]
# soglia_ritmicità=0.2 #dato un periodo le distanze tra i picchi devono discostarsi in media al più di soglia_ritmicità*periodo dal periodo
# soglia_similarità_picchi=0.5 #i 2 neuroni devono avere un coefficente di variazione della preminenza normalizzata < di soglia_similarità_picchi
# soglia_preminenza=0.3 #i 2 neuroni devono avare una preminenza normalizzata >di soglia_preminenza
# for j in range(3,10):
#     fasi = fase_picchi_2[np.logical_and(n_picchi == j, picchi_per_int_find)]
#     bp_aux=np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)]
#     for i in range(fasi.__len__()):
#         for l in range(i+1,fasi.__len__()):
#             corr_fasi.append([j,np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j)),(np.array(fasi[i])-np.array(fasi[l])).mean()/((t_final_analysis-t_initial_analysis)/(10*j))])
#
#             #(dev standard della differenza in fase/ periodo)<soglia,preminenza pesata minore di doglia
#             #if np.logical_and(np.logical_and(np.logical_and(j>1,(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j))<0.3),pr_mean_2_pesato[i]>0.4),pr_mean_2_pesato[l]>0.4):
#             #    corr_fasi_buone2.append([j,np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i])-np.array(fasi[l])).std()/((t_final_analysis-t_initial_analysis)/(10*j)),(np.array(fasi[i])-np.array(fasi[l])).mean()/((t_final_analysis-t_initial_analysis)/(10*j))])
#             preminenza_sopra_soglia1=pr_mean_2_pesato[np.where(np.array(bp)==bp_aux[i])][0]>soglia_preminenza
#             preminenza_sopra_soglia2 = pr_mean_2_pesato[np.where(np.array(bp) == bp_aux[l])][0]>soglia_preminenza
#             if np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(abs(np.array(fasi[i][1:])-np.array(fasi[i][:-1])).max()/((t_final_analysis-t_initial_analysis)/(10*j))<soglia_ritmicità,abs(np.array(fasi[l][1:])-np.array(fasi[l][:-1])).max()/((t_final_analysis-t_initial_analysis)/(10*j))<soglia_ritmicità),CV_pr_2[i]<soglia_similarità_picchi),CV_pr_2[l]<soglia_similarità_picchi),preminenza_sopra_soglia1),preminenza_sopra_soglia2):
#                 corr_fasi_buone.append([j, np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][i],np.array(bp)[np.logical_and(n_picchi == j, picchi_per_int_find)][l],(np.array(fasi[i]) - np.array(fasi[l])).std() / ((t_final_analysis - t_initial_analysis) / (10 * j)),(np.array(fasi[i]) - np.array(fasi[l])).mean() / ((t_final_analysis - t_initial_analysis) / (10 * j))])
#
# # i=0
# # fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# # fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
# # pr_mean_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# # pr_mean_2[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
# #
# # pr_mean_2_pesato[np.where(np.array(bp)==corr_fasi_buone[i][1])][0]
# # pr_mean_2_pesato[np.where(np.array(bp)==corr_fasi_buone[i][2])][0]
# # posizioni_centri[id_center[corr_fasi_buone[i][1]]]
# # posizioni_centri[id_center[corr_fasi_buone[i][2]]]
# # np.array(fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0])-np.array(fase_picchi_2[np.where(np.array(bp)==corr_fasi_buone[i][1])][0])
# # #posizioni dei centri di blocchi con firing correlati
# # posizioni_centri[id_center[corr_fasi_buone[i][1]]]
# # posizioni_centri[id_center[corr_fasi_buone[i][2]]]
#
#
# for i in range(block_plotted.__len__()):
#     #col[i]=min(15,int(pr_mean[i]/pr_mean.max() * col_max))
#     col[i] = int(pr_mean[i] / pr_mean.max() * col_max)

#
# line_x=[]
# line_y=[]
# line_z=[]
#
# for i in range(corr_fasi_buone.__len__()):
#     line_x.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][0])
#     line_x.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][0])
#     line_x.append(None)
#
#     line_y.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][1])
#     line_y.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][1])
#     line_y.append(None)
#
#     line_z.append(posizioni_centri[id_center[corr_fasi_buone[i][1]]][2])
#     line_z.append(posizioni_centri[id_center[corr_fasi_buone[i][2]]][2])
#     line_z.append(None)
#
# fig9 = go.Figure(data=points, layout=layout)
# fig9.add_scatter3d(x=line_x,
#                    y=line_y,
#                    z=line_z,
#                    mode='lines',
#                    marker=dict(color="green", size=1), opacity=0.7)
#
# for i in range(corr_fasi_buone.__len__()):
#     #if corr_fasi_buone[i][-1]>0:
#     if np.logical_or(np.logical_and(corr_fasi_buone[i][-1]>0,0.5>corr_fasi_buone[i][-1]),-0.5>corr_fasi_buone[i][-1]):
#         fig9.add_cone(x=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][0]],
#                             y=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][1]],
#                             z=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][2]],
#                             u=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][0] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][0]],
#                             v=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][1] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][1]],
#                             w=[posizioni_centri[id_center[corr_fasi_buone[i][1]]][2] - posizioni_centri[id_center[corr_fasi_buone[i][2]]][2]],
#                             sizemode="absolute",
#                             sizeref=30,
#
#                             anchor="tip"
#                             )
#     else:
#         fig9.add_cone(x=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0]],
#                       y=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1]],
#                       z=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2]],
#                       u=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][0] -posizioni_centri[id_center[corr_fasi_buone[i][1]]][0]],
#                       v=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][1] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][1]],
#                       w=[posizioni_centri[id_center[corr_fasi_buone[i][2]]][2] - posizioni_centri[id_center[corr_fasi_buone[i][1]]][2]],
#                       sizemode="absolute",
#                       sizeref=30,
#
#                       anchor="tip"
#                       )
#
# fig9.write_html(results_path + "prominence_" + sigla + "_min_" + str(min_spk) + "_min_neu_Xb_" + str(
#         neu_min_per_blocco_to_plot) + ".html")
#
#
# #color_discrete_sequence = ["black","orange", "red", "green", "blue", "pink","yellow"]
# color_discrete_sequence=px.colors.qualitative.Alphabet
# id=1
# n_predecessori=2