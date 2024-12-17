import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import h5py
import csv
import pandas as pd
import sys
import json


add_clique=False

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

points = go.Scatter3d(x=f_pos[pos_neuron_list[10]][:, 1],
                      y=f_pos[pos_neuron_list[10]][:, 2],
                      z=f_pos[pos_neuron_list[10]][:, 3],
                      mode='markers',
                      marker=dict(size=1,
                                  color='gray',
                                  showscale=True, opacity=0.1)
                      ,

                      )
camera = dict(eye=dict(x=2, y=0, z=0))
layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=10),
                   scene_camera=camera)

#fig2 = go.Figure( data=points,layout=layout)#data=points,
fig2 = go.Figure( layout=layout)#data=points,
# lightsteelblue, lightyellow, lime, limegreen,
# linen, magenta, maroon, mediumaquamarine,
# mediumblue, mediumorchid, mediumpurple,
# mediumseagreen, mediumslateblue, mediumspringgreen,
# mediumturquoise, mediumvioletred, midnightblue,
# mintcream, mistyrose, moccasin, navajowhite, navy,
# oldlace, olive, olivedrab, orange, orangered,
# orchid, palegoldenrod, palegreen, paleturquoise,
# palevioletred, papayawhip, peachpuff, peru, pink,
# plum, powderblue, purple, red, rosybrown,
# royalblue, saddlebrown, salmon, sandybrown,
# seagreen, seashell, sienna, silver, skyblue,
# slateblue, slategray, slategrey, snow, springgreen,
# steelblue, tan, teal, thistle, tomato, turquoise,
# violet, wheat, white, whitesmoke, yellow,yellowgreen

#color_discrete_sequence = ["orange", "red","red", "green","green", "blue", "blue", "pink", "pink","purple","yellow","yellow","olive","springgreen","purple","moccasin","orange","mediumblue", "mediumorchid", "mediumpurple","saddlebrown", "salmon", "sandybrown"," paleturquoise"]
for n in range(1,11):
    print(n)
    if n==10:
        list_of_neuron = np.where(np.isin(np.arange(1,261843), aux, invert=True))
    else:
        list_of_neuron = pd.read_csv(work_path+"altaAttivabilitaDisgiunti"+str(n)+"_id.csv", header=None)
    #spk_list=np.array(list_of_list_spk)
    if n==1:
        aux=np.array(list_of_neuron[0])
        colori=np.ones(list_of_neuron[0].__len__())*n/10
    else:
        if n == 10:
            aux = np.append(aux, np.array(list_of_neuron[0]+1))
        else:
            aux=np.append(aux,np.array(list_of_neuron[0]))
        colori=np.append(colori,np.ones(list_of_neuron[0].__len__())*n/10%1)
    #posizioni=[]
    #for i in aux:
        #print(i)
    #    posizioni.append(f_pos[pos_neuron_list[10]][i-1,1:])

pos=aux.argsort()
posizioni=f_pos[pos_neuron_list[10]][aux[pos] - 1, 1:]
fig2.add_scatter3d(x = np.array(posizioni)[:,0],
                    y = np.array(posizioni)[:,1],
                    z = np.array(posizioni)[:,2],
                    mode = 'markers',
                    marker = dict( size = 1,
                                   colorscale='pinkyl',
                                   color=colori[pos],#np.random.rand(1)[0]n,#color_discrete_sequence[n],#n/10,
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
fig2.write_html(work_path+"scatter_all_2.html")

if add_clique:

    if sys.argv.__len__()>2:
        work_path = sys.argv[1]
        sigla = sys.argv[2]
        setting_file = sys.argv[3]
        sim_conf = json.load(open('%s' % (setting_file), 'r'))

    else:
        work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
        #work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
        #work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
        sigla = "sl9"
        #sigla="ls5"
        #sigla = "bkg_5hz"
        setting_file = "./configuration.json"
        sim_conf = json.load(open('%s' % (setting_file), 'r'))
    t_initial_analysis = sim_conf['start_time']
    t_final_analysis = sim_conf['end_time']  # 0#5000#
    interval_dim = sim_conf['bins_dimension']
    n_workers = -1
    perc_attivi = sim_conf['percentual_of_firing_bins_for_active']
    soglia_attivi = perc_attivi * (t_final_analysis - t_initial_analysis) / interval_dim
    perc_corr = sim_conf['percentual_of_egual_bins_for_correlation']
    soglia_di_correlazione = perc_corr * (t_final_analysis - t_initial_analysis) / interval_dim
    n_shift = sim_conf['n_shift']
    n_neuroni=288027
    n_neuroni_max_da_selezionare = sim_conf['n_of_neurons_max_to_select']  # 10000#288027
    n_neuroni_min_comp_connessa = sim_conf['n_of_neurons_min_for_connected_component']
    min_clique_size = sim_conf['n_of_neurons_min_for_clique']

    print(work_path)

    results_path=work_path+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
    results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"
    with open(results_sub_path + 'clique_info.pkl', 'rb') as f:
        [id_neu_cliques, indici_cl, cl_spk, col, Isi_cliques, perc_pyr] = pickle.load(f)

    posizioni=[]
    for i in  range(len(pos_neuron_list)):
        posizioni.append(f_pos[pos_neuron_list[i]][:])

    posizioni_neuroni=posizioni[0]
    for j in range(1,posizioni.__len__()):
        posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))

    color_discrete_sequence = ["orange", "red", "green",  "pink","blue", "pink","yellow","yellow","olive","springgreen","purple","moccasin","orange","mediumblue", "mediumorchid", "mediumpurple","saddlebrown", "salmon", "sandybrown"," paleturquoise"]

    for l in np.nditer(np.array(indici_cl)):#range(id_neu_cliques.__len__()):

        neuron_to_plt=np.in1d(posizioni_neuroni[:,0],id_neu_cliques[l])
        fig2.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                            y=posizioni_neuroni[neuron_to_plt,2],
                            z=posizioni_neuroni[neuron_to_plt,3],
                            mode='markers',
                            name='clique_' + str(l),
                            marker=dict(size=5,
                                        #colorscale='tempo',
                                        color=color_discrete_sequence[l],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                        )
                            )
        i=i+1



    fig2.write_html("scatter_all"+sigla+"2.html")

