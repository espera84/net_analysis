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


t_initial_analysis=0
t_final_analysis=10000#0#5000#
interval_dim=100
n_sim=2

n_bin=int((t_final_analysis-t_initial_analysis)/interval_dim)

n_com_nrn=np.zeros([n_sim,n_bin],dtype=int)
n_fr_nrn=np.zeros([n_sim,n_bin],dtype=int)
time=np.zeros(n_bin,dtype=int)
spk_nrn=np.empty([n_sim,n_bin],dtype=object)

work_path=np.empty(n_sim,dtype=object)
spk_lists=np.empty(n_sim,dtype=object)


work_path[0]="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl2-1000_p2p5_i2i1_i2p5_p2i1_263e84f9-50a6-4a90-80fb-75a0017c89ed/"
work_path[1]="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl15-1000_p2p5_i2i1_i2p5_p2i1_78b7584c-a27f-4e30-97cd-9345d5417efc/"
#work_path[0]="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
#work_path[1]="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
for i in range(n_sim):
    list_of_list_spk=h5py.File(work_path[i]+"activity_network.hdf5", "r")
    spk_lists[i]=list_of_list_spk['spikes'].astype('float64')[:]

i=0

for t in range(t_initial_analysis,t_final_analysis,interval_dim):
    print(t)
    #for j in range(n_sim):
    spk_nrn[0][i]=spk_lists[0][np.logical_and(t<spk_lists[0][:,1], spk_lists[0][:,1] <(t+interval_dim)),0]
    spk_nrn[1][i] = spk_lists[1][np.logical_and(t<spk_lists[1][:,1], spk_lists[1][:,1] <(t+interval_dim)),0]

    n_com_nrn[0][i]=np.unique(spk_nrn[0][i][np.in1d(spk_nrn[0][i], spk_nrn[1][i])]).__len__()
    n_fr_nrn[0][i]=np.unique(spk_nrn[0][i]).__len__()
    n_com_nrn[1][i] = np.unique(spk_nrn[1][i][np.in1d(spk_nrn[1][i], spk_nrn[0][i])]).__len__()
    n_fr_nrn[1][i] = np.unique(spk_nrn[1][i]).__len__()
    time[i]=t+interval_dim

    i = i + 1


plt.figure()
plt.plot(time,n_com_nrn[0][:]/n_fr_nrn[0][:], label= "sim sl2 and Bkg5hz")
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('% of common firing neurons')
plt.show()


plt.figure()
plt.plot(time,n_com_nrn[1][:]/n_fr_nrn[1][:], label= "sim sl2 and Bkg5hz")
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('% of common firing neurons')
plt.show()
try:
    os.mkdir("C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/compare/")
except:
    pass

plt.savefig("C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/compare/Asl2_Bkg5hz_int_"+str(interval_dim)+".png")