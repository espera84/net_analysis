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

n_sim=3

work_path=np.empty(n_sim,dtype=object)
sigla=np.empty(n_sim,dtype=object)
id_neu_cliques=np.empty(n_sim,dtype=object)
indici_cl=np.empty(n_sim,dtype=object)
results_path=np.empty(n_sim,dtype=object)
results_sub_path=np.empty(n_sim,dtype=object)
is_common_neuron=np.empty([n_sim,n_sim],dtype=object)
perc_common_neuron=np.empty([n_sim,n_sim],dtype=object)

work_path[0] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
sigla[0]="sl9"
work_path[1] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
sigla[1]="ls5"
work_path[2] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
sigla[2]="bkg_5hz"




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
for i in range(n_sim):
    results_path[i]=work_path[i]+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
    results_sub_path[i] = results_path[i] + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"
    with open(results_sub_path[i] + 'clique_info.pkl', 'rb') as f:
        [id_neu_cliques[i], indici_cl[i], cl_spk, col, Isi_cliques, perc_pyr] = pickle.load(f)
    print("connected components data loaded")



#n_coppie=n*n-1
percentage_of_common_neuron=[]
mean_percentage_for_cliques=[]
n_clique_to_compare=4
for k in range(n_clique_to_compare):
    for i in range(n_sim):
        for j in range(n_sim):
            is_common_neuron[i][j]=np.in1d(id_neu_cliques[i][indici_cl[i][k]],id_neu_cliques[j][indici_cl[j][k]])
            perc_common_neuron[i][j]=is_common_neuron[i][j].sum()/is_common_neuron[i][j].__len__()
    percentage_of_common_neuron.append(perc_common_neuron.copy())

    mean_percentage_for_cliques.append(percentage_of_common_neuron[k][percentage_of_common_neuron[k]!=1].mean())