import numpy as np
import h5py
from joblib import Parallel, delayed
import time
import os
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly
import math
import networkx as nx
from joblib import Parallel, delayed
import json
import sys


def calcola_pr_in_uscita(p,n_ex,n_in):

    P_Combinazioni_pos_tot=0
    P_Combinazioni_pos=0
    for k in range (n_in+n_ex):
        for i in range(int(np.floor(k/2))+1,min(k,n_ex)):

            if (k-n_in>=i) and i<=k:
                #print("enter")
                aux=math.pow(p,k)*math.pow(1-p,n_ex+n_in-k)* math.comb(n_ex, i)*math.comb(n_in, k-i)
                #print(aux)
                P_Combinazioni_pos = P_Combinazioni_pos +  math.comb(n_ex, i)*math.comb(n_in, k-i)
        print("k prob",k,P_Combinazioni_pos)
        P_Combinazioni_pos_tot = P_Combinazioni_pos_tot+math.pow(p,k)*math.pow(1-p,n_ex+n_in-k)*P_Combinazioni_pos
    return  P_Combinazioni_pos_tot

def calcola_pr_in_uscita(id_pyr):
    print(id_pyr)
    Combinazioni_tot=0
    Combinazioni_pos=0
    for i in range (n_con_tot_on_pyr[id_pyr]):
        Combinazioni_tot=Combinazioni_tot+math.comb(n_con_tot_on_pyr[id_pyr], i)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for j in range(int(np.floor(i/2))+1,min(i,n_con_ex_on_pyr[id_pyr])):
            Combinazioni_pos = Combinazioni_pos + math.comb(n_con_ex_on_pyr[id_pyr], j)*math.comb(n_con_in_on_pyr[id_pyr], i-j)


    return  [Combinazioni_pos/Combinazioni_tot,Combinazioni_pos,Combinazioni_tot]

def calcola_pr_firing(id_neuorn,fat_molt):
    global n_con_tot_on,n_con_ex_on

    print(id_neuorn)
    Combinazioni_tot=0
    Combinazioni_pos=0
    for k in range (int(np.floor(n_con_tot_on[id_neuorn]/10)),n_con_tot_on[id_neuorn]):
        #quante sono le combinazioni in ingresso di k stimoli sul neurone id_pyr (per k>di un decimo del numero di connessioni totali (eccitatorie e inibitorie)
        Combinazioni_tot=Combinazioni_tot+math.comb(n_con_tot_on[id_neuorn], k)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for i in range(int(np.floor(k*fat_molt))+1,min(k,n_con_ex_on[id_neuorn])): #almeno fat_molt delle k sinapsi devono essere eccitatorie
            if (k - i< n_con_in_on[id_pyr]): #verifico che ci siano almeno k-i syn inibitorie
                Combinazioni_pos = Combinazioni_pos + math.comb(n_con_ex_on[id_neuorn], i)*math.comb(n_con_in_on[id_neuorn], k-i)


    return  [Combinazioni_pos/Combinazioni_tot,Combinazioni_pos,Combinazioni_tot]


def calcola_pr_firing_dif(id_neuorn,fat_add,n_con_tot_on,n_con_ex_on,n_con_in_on):


    #print(id_pyr)
    Combinazioni_tot=0
    Combinazioni_pos=0
    for k in range (fat_add,n_con_tot_on[id_neuorn]): #k è il numero di sinapsi ipoteticamente attive sul neurone tra eccitatorie e inibitorie
        #quante sono le combinazioni in ingresso di k stimoli sul neurone id_pyr (per k>di un decimo del numero di connessioni totali (eccitatorie e inibitorie)
        Combinazioni_tot=Combinazioni_tot+math.comb(n_con_tot_on[id_neuorn], k)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for j in range(fat_add, k):
            #for l in range(j - fat_add + 1):#il numero di sinapsi eccitatorie in input devono essere almeno fat_add in più delle l sinapsi inibitorie
            #    print(j, l,k-j)
            if (k - j < n_con_in_on[id_neuorn]):  # verifico che ci siano almeno k-j syn inibitorie
                Combinazioni_pos = Combinazioni_pos + math.comb(n_con_ex_on[id_neuorn], j) * math.comb(n_con_in_on[id_neuorn], k - j)

    try:
        rat=Combinazioni_pos/Combinazioni_tot
    except:
        rat=0

    return  [rat,Combinazioni_pos,Combinazioni_tot]


def calcola_pr_firing_dif_2gen(id_pyr,fat_add):
    global n_con_tot_on_pyr,n_con_ex_on_pyr

    #print(id_pyr)
    Combinazioni_tot=0
    Combinazioni_pos=0
    for k in range (fat_add,n_con_tot_on_pyr[id_pyr]): #k è il numero di sinapsi ipoteticamente attive sul neurone tra eccitatorie e inibitorie
        #quante sono le combinazioni in ingresso di k stimoli sul neurone id_pyr (per k>di un decimo del numero di connessioni totali (eccitatorie e inibitorie)
        Combinazioni_tot=Combinazioni_tot+math.comb(n_con_tot_on_pyr[id_pyr], k)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for j in range(fat_add, k):
            #for l in range(j - fat_add + 1):#il numero di sinapsi eccitatorie in input devono essere almeno fat_add in più delle l sinapsi inibitorie
            #    print(j, l,k-j)
            if (k - j < n_con_in_on_pyr[id_pyr]):  # verifico che ci siano almeno k-j syn inibitorie
                Combinazioni_pos = Combinazioni_pos + math.comb(n_con_ex_on_pyr[id_pyr], j) * math.comb(n_con_in_on_pyr[id_pyr], k - j)

    try:
        rat=Combinazioni_pos/Combinazioni_tot
    except:
        rat=0

    return  [rat,Combinazioni_pos,Combinazioni_tot]

def calcola_prob_2g(fat_add):
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
    n_fam_neu=pos_neuron_list.__len__()

    modalità="additiva"
    add_clique=True
    plot_only_int=True

    N_pyr=261843
    fat_molt=2/3;
    #fat_add=200
    path='C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python\\'

    with open('Connection_info_add_' + str(fat_add) +'_prob_2_gen.pkl', 'rb') as f:
        [id, n_con_ex_on, n_con_in_on] = pickle.load(f)

    #with open( 'Pyr_connection_info_out.pkl', 'rb') as f:
    #    [id, n_con_ex_on_pyr_out, n_con_in_on_pyr_out] = pickle.load(f)

    try:
        if modalità=="additiva":
            with open('Firing_Probability_all_add_'+str(fat_add)+'_2_gen.pkl', 'rb') as f:
                [id,probabilità_firing, Combinazioni_tot, Combinazioni_pos] = pickle.load(f)
        else:
            with open('Firing_Probability_all_mul_'+str(fat_molt)+'_2_gen.pkl', 'rb') as f:
                [id,probabilità_firing, Combinazioni_tot, Combinazioni_pos] = pickle.load(f)
    except:

        N_neuroni=id.__len__()

        n_con_ex_on=n_con_ex_on.astype(int)
        n_con_in_on=n_con_in_on.astype(int)
        #n_con_ex_on_out=n_con_ex_on_out.astype(int)
        #n_con_in_on_out=n_con_in_on_out.astype(int)


        n_con_tot_on=n_con_ex_on+n_con_in_on
        #n_con_tot_on_out=n_con_ex_on_pyr_out+n_con_in_on_pyr_out

        n_workers=os.cpu_count()


    #    Ltt = Parallel(n_jobs=n_workers, verbose=50)(delayed(calcola_pr_firing)(i,fat_molt) for i in range(N_pyr)) #prob firing considerando il rapporto tra sin eccitatorie e sin inibitorie
        Ltt = Parallel(n_jobs=n_workers, verbose=50)(delayed(calcola_pr_firing_dif)(i, fat_add,n_con_tot_on,n_con_ex_on,n_con_in_on) for i in range(N_neuroni)) #prob firing considerando la differenza tra sin eccitatorie e sin inibitorie

        results = np.array(Ltt)

        probabilità_firing=results[:,0]
        Combinazioni_pos =results[:,1]
        Combinazioni_tot =results[:,2]
        id=np.arange(1,N_neuroni+1,1)
        if modalità=="additiva":
            with open('Firing_Probability_all_add_'+str(fat_add)+'_2_gen.pkl', 'wb') as f:
                pickle.dump([id, probabilità_firing, Combinazioni_tot, Combinazioni_pos], f)
        else:
            with open(path+'Firing_Probability_all_mul_'+str(fat_molt)+'_2_gen.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([id,probabilità_firing, Combinazioni_tot, Combinazioni_pos], f)

        counts, bins = np.histogram(probabilità_firing, 100)
        plt.stairs(counts, bins)
        plt.show()

    #    probabilità_firing.max()

    #    f_pos[pos_neuron_list[10]][np.where(probabilità_firing > probabilità_firing.max() / 2)[:], 1:4]
    #    pos_neuron_with_higher_prob=f_pos[pos_neuron_list[10]][np.where(probabilità_firing>probabilità_firing.max()/2)[0],1:4]




        #
        #
        # points = go.Scatter3d(x=pos_neuron_with_higher_prob[:, 0],
        #                       y=pos_neuron_with_higher_prob[:, 1],
        #                       z=pos_neuron_with_higher_prob[:, 2],
        #                       mode='markers',  # 'lines+markers',
        #                       marker=dict(size=1,
        #                                   showscale=True),
        #
        #                       )

    layout = go.Layout(title='num neurons with high probability of firing',
                           margin=dict(l=0,
                                       r=0,
                                       b=0,
                                       t=0)
                           )
    fig2 = go.Figure(layout=layout)  # data=points,

    #posizioni=f_pos[pos_neuron_list[10]][:, 1:]

    posizioni = []
    for i in range(len(pos_neuron_list)):
        posizioni.append(f_pos[pos_neuron_list[i]][:])

    posizioni_neuroni = posizioni[0]
    for j in range(1, posizioni.__len__()):
        posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))
    ordine_neu=posizioni_neuroni[:,0].argsort()
    posizioni_neu_ord=posizioni_neuroni[ordine_neu,:].copy()

    fig2.add_scatter3d(x = np.array(posizioni_neu_ord)[:N_pyr,1],
                        y = np.array(posizioni_neu_ord)[:N_pyr,2],
                        z = np.array(posizioni_neu_ord)[:N_pyr,3],
                        mode = 'markers',
                        marker = dict( size = 1,
                                       colorscale='pinkyl',
                                       color=probabilità_firing[:N_pyr],#np.random.rand(1)[0]n,#color_discrete_sequence[n],#n/10,
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
    if modalità=="additiva":
            fig2.write_html("theoretical_prob_firing_map_pyr_add_" +str(fat_add)+"_2_gen.html")
    else:
            fig2.write_html("theoretical_prob_firing_map_pyr_mult_" + str(fat_molt) +"_2_gen.html")

    fig_all=go.Figure(fig2)
    fig_all.add_scatter3d(x = np.array(posizioni_neu_ord)[N_pyr:,1],
                        y = np.array(posizioni_neu_ord)[N_pyr:,2],
                        z = np.array(posizioni_neu_ord)[N_pyr:,3],
                        mode = 'markers',
                        marker = dict( size = 1,
                                       colorscale='aggrnyl',
                                       color=probabilità_firing[N_pyr:],#np.random.rand(1)[0]n,#color_discrete_sequence[n],#n/10,
                                       showscale=True)
                        )  # ,showscale=True), )



    if modalità=="additiva":
            fig_all.write_html("theoretical_prob_firing_map_all_add_" +str(fat_add)+"_2_gen.html")
    else:
            fig_all.write_html("theoretical_prob_firing_map_all_mult_" + str(fat_molt) + "_2_gen.html")
    #fig2.show()


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
            fig_all.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                                y=posizioni_neuroni[neuron_to_plt,2],
                                z=posizioni_neuroni[neuron_to_plt,3],
                                mode='markers',
                                name='clique_' + str(l),
                                marker=dict(size=3,
                                            #colorscale='tempo',
                                            color=color_discrete_sequence[l],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                            )
                                )

            fig2.add_scatter3d(x=posizioni_neuroni[neuron_to_plt, 1],
                               y=posizioni_neuroni[neuron_to_plt, 2],
                               z=posizioni_neuroni[neuron_to_plt, 3],
                               mode='markers',
                               name='clique_' + str(l),
                               marker=dict(size=3,
                                           # colorscale='tempo',
                                           color=color_discrete_sequence[l],
                                           # px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                           )
                               )

        if modalità == "additiva":
            fig2.write_html("theoretical_prob_firing_map_pyr_add_" + str(fat_add) + "with_CL_"+sigla+"_2_gen.html")
            fig_all.write_html("theoretical_prob_firing_map_all_add_" + str(fat_add) + "with_CL_" + sigla + "_2_gen.html")
        else:
            fig2.write_html("theoretical_prob_firing_map_pyr_add_" + str(fat_add) + "with_CL_"+sigla+"_2_gen.html")
            fig_all.write_html("theoretical_prob_firing_map_all_add_" + str(fat_add) + "with_CL_" + sigla + "_2_gen.html")
       #fig2.write_html("theoretical_prob_firing_map_mult_" + str(fat_add) + "with_CL_"+sigla+"2.html")
    if plot_only_int:
        #fig2.show()

        fig2 = go.Figure(layout=layout)  # data=points,

        fig2.add_scatter3d(x = np.array(posizioni_neu_ord)[N_pyr:,1],
                            y = np.array(posizioni_neu_ord)[N_pyr:,2],
                            z = np.array(posizioni_neu_ord)[N_pyr:,3],
                            mode = 'markers',
                            marker = dict( size =2,
                                       colorscale='aggrnyl',
                                       color=probabilità_firing[N_pyr:],#np.random.rand(1)[0]n,#color_discrete_sequence[n],#n/10,
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

        if add_clique:

            for l in np.nditer(np.array(indici_cl)):#range(id_neu_cliques.__len__()):

                neuron_to_plt=np.in1d(posizioni_neuroni[:,0],id_neu_cliques[l])
                fig2.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                                   y=posizioni_neuroni[neuron_to_plt,2],
                                   z=posizioni_neuroni[neuron_to_plt,3],
                                   mode='markers',
                                   name='clique_' + str(l),
                                   marker=dict(size=5,
                                       #colorscale='tempo',
                                       color=color_discrete_sequence[l],  #px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                       )
                                   )
            if modalità == "additiva":
                fig2.write_html("theoretical_prob_firing_map_int_add_" + str(fat_add) + "with_CL_" + sigla + "_2_gen.html")
            else:
                fig2.write_html("theoretical_prob_firing_map_int_mult_" + str(fat_molt) + "with_CL_" + sigla + "_2_gen.html")
