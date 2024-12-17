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

path='C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python\\'
with open(path+'Pyr_connection_info.pkl', 'rb') as f:
    [id, n_con_ex_on_pyr, n_con_in_on_pyr] = pickle.load(f)

with open(path + 'Pyr_connection_info_out.pkl', 'rb') as f:
    [id, n_con_ex_on_pyr_out, n_con_in_on_pyr_out] = pickle.load(f)

N_pyr=id.__len__()

n_con_ex_on_pyr=n_con_ex_on_pyr.astype(int)
n_con_in_on_pyr=n_con_in_on_pyr.astype(int)
n_con_ex_on_pyr_out=n_con_ex_on_pyr_out.astype(int)
n_con_in_on_pyr_out=n_con_in_on_pyr_out.astype(int)


n_con_tot_on_pyr=n_con_ex_on_pyr+n_con_in_on_pyr
n_con_tot_on_pyr_out=n_con_ex_on_pyr_out+n_con_in_on_pyr_out
Combinazioni_tot=np.zeros((N_pyr,),dtype=object)
Combinazioni_pos=np.zeros((N_pyr,),dtype=object)
Combinazioni_tot_out=np.zeros((N_pyr,),dtype=object)
Combinazioni_pos_out=np.zeros((N_pyr,),dtype=object)
n_con_tot_on_pyr=n_con_tot_on_pyr.astype(int)
n_con_tot_on_pyr_out=n_con_tot_on_pyr_out.astype(int)
n_workers=os.cpu_count()

for id_pyr in range(N_pyr):
    print(id_pyr)
    for i in range (n_con_tot_on_pyr[id_pyr]):
        Combinazioni_tot[id_pyr]=Combinazioni_tot[id_pyr]+math.comb(n_con_tot_on_pyr[id_pyr], i)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for j in range(int(np.floor(i/2))+1,min(i,n_con_ex_on_pyr[id_pyr])):
            Combinazioni_pos[id_pyr] = Combinazioni_pos[id_pyr] + math.comb(n_con_ex_on_pyr[id_pyr], j)*math.comb(n_con_in_on_pyr[id_pyr], i-j)

#        Combinazioni_tot_out[id_pyr] = Combinazioni_tot_out[id_pyr] + math.comb(n_con_tot_on_pyr_out[id_pyr], i)
        # combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
#        for j in range(int(np.floor(i/2))+ 1, min(i, n_con_ex_on_pyr_out[id_pyr])):
#            Combinazioni_pos_out[id_pyr] = Combinazioni_pos_out[id_pyr] + math.comb(n_con_ex_on_pyr_out[id_pyr],j) * math.comb(n_con_in_on_pyr_out[id_pyr], i - j)

P_in=Combinazioni_pos/Combinazioni_tot
#P_out=Combinazioni_pos_out/Combinazioni_tot_out
