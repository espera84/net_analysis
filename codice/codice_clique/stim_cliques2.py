import numpy as np
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

def find_stimulation_neurons3(CL):

    lung = clusters_diff[ind].__len__()
    if lung != 0:
        n_cl = int(clusters_diff[ind][:, 5].max()) + 1
        id_neu_che_potenzialmente_stimolano = np.empty((n_cl,), dtype=object)

        for i in range(n_cl):
            ind_neu_cluster = clusters_diff[ind][np.logical_and(clusters_diff[ind][:, 0] <len(in_conn),clusters_diff[ind][:, 5] == i), 0].astype(int)
            out = in_conn[ind_neu_cluster]
            out2=[]
            for k in range(out.__len__()):
                #print(k)
                out2 = np.unique(out2 + out[k]).tolist()
            id_neu_che_potenzialmente_stimolano[i] = np.unique(out2)


        return id_neu_che_potenzialmente_stimolano
    else:
        return np.empty(0)




def save_raster_all(list_id_neuroni):

    global t_initial_analysis,t_final_analysis
    spk = []

    color = np.array([])
    plt.figure()
    pos_n=1
    for j in range(list_id_neuroni.__len__()):
        id_neuroni=list_id_neuroni[j]
        first_neuron=True
        print("lista", j)
        print("lung lista",id_neuroni.__len__())
        for id_neuron in np.nditer(id_neuroni):

            spk_id=spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]
            spk_id_interval=spk_id[np.logical_and(spk_id>t_initial_analysis , spk_id<t_final_analysis)]
            color=np.concatenate((color,np.ones(spk_id_interval.__len__())*j))
            #plt.scatter(spk_list[pos_spk_id, 1][0],i*np.ones(spk_list[pos_spk_id, 1][0].__len__()))
            #spk.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][np.logical_and(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]>t_initial_analysis , spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]<t_final_analysis)])
            spk.append(spk_id_interval)


    for i in range(spk.__len__()):
        if i == 0:
            #cl_spk = np.vstack((np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__()))), np.ones(spk[i].__len__()) * i))
            cl_spk = np.vstack((spk[i], np.ones(spk[i].__len__()) * i))
            #cl_spk = np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))
        else:
            # print(np.ones(spk[i].__len__()))
            #cl_spk = np.concatenate((cl_spk, np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))), axis=1)
            cl_spk = np.concatenate((cl_spk,np.vstack((spk[i], np.ones(spk[i].__len__()) * i))) , axis=1)
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
        spk_id=spk_list[np.where(np.in1d(spk_list[:, 0], list_id_neuroni[j]))[0],1]
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




current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
parent_path = os.path.dirname(parent_path)

data_net_path=parent_path+"/input_data/data_net/"

setting_file = "./clique_configuration.json"
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
sim_name=sim_conf["folder_simulation_name"]#"sl9_9-1-9-1"

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

results_path=results_path+"/clique_interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"

with open(results_sub_path + 'clique_info.pkl', 'rb') as f:
    [id_neu_cliques, indici_cl, cl_spk, col, Isi_cliques, perc_pyr] = pickle.load(f)
print("connected components data loaded")


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

posizioni=[]
for i in range(len(pos_neuron_list)):
    posizioni.append(f_pos[pos_neuron_list[i]][:])

posizioni_neuroni=posizioni[0]
for j in range(1,posizioni.__len__()):
    posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))
try:
    with open(results_path+'stim_clique_info_.pkl', 'rb') as f:
        [neuron_stim_pyr, neuron_stim_int, cl_spk_int, col_int, cl_spk_pyr, col_pyr,n_stim_pyr,n_stim_int] = pickle.load(f)


    print("stim neurons loaded")
except:
    neuron_stim_pyr = np.empty(indici_cl.__len__(), dtype=object)
    neuron_stim_int = np.empty(indici_cl.__len__(), dtype=object)
    n_stim_pyr=np.zeros(indici_cl.__len__())
    n_stim_int = np.zeros(indici_cl.__len__())
    for i_cl in indici_cl:
        aux = f_pyr[pyr_connection_list[0]][np.isin(f_pyr[pyr_connection_list[0]][:, 1], id_neu_cliques[indici_cl[i_cl]]), 0]
        n_stim_pyr[i_cl]=n_stim_pyr[i_cl]+aux.__len__()
        neuron_stim_pyr[i_cl] = np.unique(aux)


        for j in range(in_connection_list.__len__()):
            if j==0:
                neuron_stim_int[i_cl] =[]
            if in_connection_list[j][:5] == "SP_PC":
                aux = f_in[in_connection_list[j]][np.isin(f_in[in_connection_list[j]][:, 1],id_neu_cliques[indici_cl[i_cl]]), 0]
                n_stim_pyr[i_cl] = n_stim_pyr[i_cl] + aux.__len__()
                neuron_stim_pyr[i_cl]=np.concatenate((neuron_stim_pyr[i_cl],np.unique(aux)))
            else:
                aux=f_in[in_connection_list[j]][np.isin(f_in[in_connection_list[j]][:, 1],id_neu_cliques[indici_cl[i_cl]]), 0]
                n_stim_int[i_cl] = n_stim_int[i_cl] + aux.__len__()
                neuron_stim_int[i_cl]=np.concatenate((neuron_stim_int[i_cl],np.unique(aux)))

        neuron_stim_pyr[i_cl]=np.unique(neuron_stim_pyr[i_cl])
        neuron_stim_int[i_cl] = np.unique(neuron_stim_int[i_cl])



    t5=time.time()





    t4 = time.time()
    [cl_spk_int, col_int] = save_raster_all2(neuron_stim_int)
    t5 = time.time()
    [cl_spk_pyr,col_pyr]=save_raster_all2(neuron_stim_pyr)

    with open(results_path+'stim_clique_info_.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([neuron_stim_pyr, neuron_stim_int, cl_spk_int, col_int, cl_spk_pyr, col_pyr,n_stim_pyr,n_stim_int] , f)

    #aliceblue, antiquewhite, aqua, aquamarine, azure,
                # beige, bisque, black, blanchedalmond, blue,
                # blueviolet, brown, burlywood, cadetblue,
                # chartreuse, chocolate, coral, cornflowerblue,
                # cornsilk, crimson, cyan, darkblue, darkcyan,
                # darkgoldenrod, darkgray, darkgrey, darkgreen,
                # darkkhaki, darkmagenta, darkolivegreen, darkorange,
                # darkorchid, darkred, darksalmon, darkseagreen,
                # darkslateblue, darkslategray, darkslategrey,
                # darkturquoise, darkviolet, deeppink, deepskyblue,
                # dimgray, dimgrey, dodgerblue, firebrick,
                # floralwhite, forestgreen, fuchsia, gainsboro,
                # ghostwhite, gold, goldenrod, gray, grey, green,
                # greenyellow, honeydew, hotpink, indianred, indigo,
                # ivory, khaki, lavender, lavenderblush, lawngreen,
                # lemonchiffon, lightblue, lightcoral, lightcyan,
                # lightgoldenrodyellow, lightgray, lightgrey,
                # lightgreen, lightpink, lightsalmon, lightseagreen,
                # lightskyblue, lightslategray, lightslategrey,
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


color_discrete_sequence = ["orange", "red", "green", "blue", "pink","yellow","springgreen"]
colori_2=col_pyr[np.in1d(col_pyr,indici_cl)].astype('str')
colori_2_int=col_int[np.in1d(col_int,indici_cl)].astype('str')
l=0
for i in np.nditer(np.unique(col_pyr[np.in1d(col_pyr,indici_cl)])):
    #colori[col_pyr[np.in1d(col_pyr,ind)]==i]=int(l)
    #colori_2[col_pyr[np.in1d(col_pyr, ind)] == i]=color_discrete_sequence[int(l)]
    colori_2[col_pyr[np.in1d(col_pyr, indici_cl)] == i] = color_discrete_sequence[int(l)]
    l=l+1
l=0
for i in np.nditer(np.unique(col_int[np.in1d(col_int, indici_cl)])):
    colori_2_int[col_int[np.in1d(col_int, indici_cl)] == i] = color_discrete_sequence[int(l)]
    l=l+1
    print("l 1 ",l)

    # l = 0
    y_raster = cl_spk[1,np.in1d(col,indici_cl)]
    # for i in np.nditer(np.unique(cl_spk[1,np.in1d(col,indici_cl)])):
    #     y_raster[cl_spk[1,np.in1d(col,indici_cl)] == i] = l
    #     l = l + 1

l = 0
l1 = 0
l_int=0
l1_int = 0
indici_cl2 = indici_cl.copy()
indici_cl2.sort()
camp=50
camp_int=1
y_raster_to_plot=np.array([])
cl_spk_to_plot=np.array([])
colori_2_to_plot=np.array([])

y_raster_to_plot_int=np.array([])
cl_spk_to_plot_int=np.array([])
colori_2_to_plot_int=np.array([])


for j in range(indici_cl2.__len__()):
    for i in np.unique(cl_spk_pyr[1, np.in1d(col_pyr, indici_cl2[j])]):
        if l%camp==0:

            # if i>N_pyr:
            #     cl_spk_to_plot_int = np.concatenate((cl_spk_to_plot_int, cl_spk_int[0, cl_spk[1, np.in1d(col_int, indici_cl)] == i]))
            #     y_raster_to_plot_int = np.concatenate((y_raster_to_plot_int, np.ones(np.sum(cl_spk_int[1, np.in1d(col_int, indici_cl)] == i)) * l1_int))
            #     #colori_2_to_plot_int = np.concatenate((colori_2_to_plot_int, colori_2[cl_spk_int[1, np.in1d(col_int, indici_cl)] == i]))
            #     l1_int = l1_int + 1
            #
            # else:
            cl_spk_to_plot=np.concatenate((cl_spk_to_plot,cl_spk_pyr[0,cl_spk_pyr[1,np.in1d(col_pyr,indici_cl)] == i ]))
            y_raster_to_plot=np.concatenate((y_raster_to_plot,np.ones(np.sum(cl_spk_pyr[1, np.in1d(col_pyr, indici_cl)] == i))*l1))
            colori_2_to_plot=np.concatenate((colori_2_to_plot,colori_2[cl_spk_pyr[1,np.in1d(col_pyr,indici_cl)] == i]))
            l1=l1+1
            print("l 1 p ", l1,l)
        l=l+1

    for i in np.unique(cl_spk_int[1, np.in1d(col_int, indici_cl2[j])]):
        if l_int % camp_int == 0:

            cl_spk_to_plot_int = np.concatenate((cl_spk_to_plot_int, cl_spk_int[0, cl_spk_int[1, np.in1d(col_int, indici_cl)] == i]))
            y_raster_to_plot_int = np.concatenate((y_raster_to_plot_int, np.ones(np.sum(cl_spk_int[1, np.in1d(col_int, indici_cl)] == i)) * l1_int))
            colori_2_to_plot_int = np.concatenate((colori_2_to_plot_int, colori_2_int[cl_spk_int[1, np.in1d(col_int, indici_cl)] == i]))
            l1_int = l1_int + 1
            print("l 1 i", l1_int,l_int)
        l_int = l_int + 1
        #print("l 2 ",l)

# for j in range(indici_cl2.__len__()):
#     for i in np.unique(cl_spk[1, np.in1d(col, indici_cl2[j])]):
#         y_raster[cl_spk[1,np.in1d(col,indici_cl)] == i] = l
#         l=l+1
#         print("l 2 ",l)

fig = go.Figure(data=go.Scatter(
    #x=cl_spk[0, np.in1d(col, indici_cl)],
    x=cl_spk_to_plot,
    y=y_raster_to_plot,

    mode='markers',

    marker=dict(
            #symbol=sim,
            symbol=142,

            size=2,
            color=colori_2_to_plot,  # set color equal to a variable
            # colorscale='pinkyl', # one of plotly colorscales
            showscale=True
    ),

))



fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.write_html( results_sub_path+"trasp_raster_pyr_stim.html")


fig = go.Figure(data=go.Scatter(
    #x=cl_spk[0, np.in1d(col, indici_cl)],
    x=cl_spk_to_plot_int,
    y=y_raster_to_plot_int,

    mode='markers',

    marker=dict(
            #symbol=sim,
            symbol=142,

            size=2,
            color=colori_2_to_plot_int,  # set color equal to a variable
            # colorscale='pinkyl', # one of plotly colorscales
            showscale=True
    ),

))



fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.write_html( results_sub_path+"trasp_raster_int_stim.html")




camera=dict(eye=dict(x=2, y=0, z=0))
points = go.Scatter3d(x=posizioni[10][0::10, 1],
                          y=posizioni[10][0::10, 2],
                          z=posizioni[10][0::10, 3],
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
                   scene_camera=camera


                       )
fig3 = go.Figure(data=points, layout=layout)
fig4=  go.Figure(data=points, layout=layout)

i=0
for l in np.nditer(np.array(indici_cl)):#range(id_neu_cliques.__len__()):

    neuron_to_plt=np.in1d(posizioni_neuroni[:,0],neuron_stim_pyr[l])
    fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                       y=posizioni_neuroni[neuron_to_plt,2],
                       z=posizioni_neuroni[neuron_to_plt,3],
                       mode='markers',
                       name='clique_' + str(l),
                       text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                       marker=dict(size=2,
                                   #colorscale='pinkyl',
                                   color=color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   #colorscale='Plotly3',
                                   #showscale=False,
                                   )
                       )

    neuron_to_plt=np.in1d(posizioni_neuroni[:,0],neuron_stim_int[l])
    fig4.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                       y=posizioni_neuroni[neuron_to_plt,2],
                       z=posizioni_neuroni[neuron_to_plt,3],
                       mode='markers',
                       name='clique_' + str(l),
                       text=posizioni_neuroni[neuron_to_plt, 0].astype('str'),
                       marker=dict(size=2,
                                   #colorscale='pinkyl',
                                   color=color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                   #colorscale='Plotly3',
                                   #showscale=False,
                                   )
                       )




    i=i+1
#fig3.show()

#fig3.update_layout(scene_camera=camera)
fig3.write_html( results_sub_path+"stim_pyr_neurons.html")
fig4.write_html( results_sub_path+"stim_int_neurons.html")
fig2=go.Figure()
fig3=go.Figure()
n_bins=int((t_final_analysis-t_initial_analysis)/interval_dim)
first_clique=True
for i in indici_cl2:
    [hist_value, bins] = np.histogram(cl_spk_pyr[0, np.in1d(col_pyr, indici_cl[i])], n_bins,[t_initial_analysis,t_final_analysis])
    if first_clique:
        first_clique=False
        fig = go.Figure(data=go.Scatter(
            # x=cl_spk[0, np.in1d(col, indici_cl)],
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value,
            name='pyr_clique_' + str(indici_cl[i]),
            #mode='markers',
            marker=dict(
                # symbol=sim,
                #symbol=142,

                size=2,
                color=color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),

        ))
    else:
        fig.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=hist_value,
            name='pyr_clique_' + str(indici_cl[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )




    print("ao")
    [hist_value_int, bins] = np.histogram(cl_spk_int[0, np.in1d(col_int, indici_cl[i])], 1000,[30000,40000])
    fig.add_scatter(
            x=(bins[1:]+bins[:-1])/2,
            y=hist_value_int,
            name='int_clique_' + str(indici_cl[i]),
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color="black",  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
                ),
        )

    ratio = hist_value / hist_value_int

    fig2.add_scatter(
            x=(bins[1:] + bins[:-1]) / 2,
            y=ratio,
            name='ratio_' + str(indici_cl[i]),
            # mode='markers',
            marker=dict(
                # symbol=sim,
                # symbol=142,

                size=2,
                color=color_discrete_sequence[i],  # set color equal to a variable
                # colorscale='pinkyl', # one of plotly colorscales
                showscale=True
            ),
        )

    dif = hist_value - hist_value_int

    fig3.add_scatter(
        x=(bins[1:] + bins[:-1]) / 2,
        y=dif,
        name='dif_' + str(indici_cl[i]),
        # mode='markers',
        marker=dict(
            # symbol=sim,
            # symbol=142,

            size=2,
            color=color_discrete_sequence[i],  # set color equal to a variable
            # colorscale='pinkyl', # one of plotly colorscales
            showscale=True
        ),
    )


fig.write_html(results_sub_path+"hist_stim_firing.html")

fig2.write_html( results_sub_path+"ratio_stim_firing.html")

fig3.write_html( results_sub_path+"dif_stim_firing.html")
#
#
# fig.show()
# plt.figure()
# plt.title('pyr cl ' + str(indici_cl[i]))
# plt.hist(cl_spk_pyr[0,np.in1d(col_pyr,indici_cl[i])],1000,color=color_discrete_sequence[i])
# plt.show()
# plt.figure()
# plt.title('int cl '+ str(indici_cl[i]))
# plt.hist(cl_spk_int[0, np.in1d(col_int, indici_cl[i])], 1000, color=color_discrete_sequence[i])
# plt.show()
#
