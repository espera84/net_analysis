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

def compute_sub_net( n_bin_x,n_bin_y,n_bin_z):

    os.chdir('C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python')

    min_connection_degree=10


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

    posizioni_interneuroni=np.empty((0,4))
    for i in range(pos_neuron_list.__len__()):
        if i==10:
            posizioni_piramidali = f_pos[pos_neuron_list[10]][:]
        else:
            posizioni_interneuroni=np.concatenate((posizioni_interneuroni,f_pos[pos_neuron_list[i]][:]))
    N_SP_PC=f_pos['SP_PC'][:,0].shape[0]


    threshold_connection=800
    path='C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python\\results_bin_x_'+str(n_bin_x)+'_bin_y_'+str(n_bin_y)+'_bin_z_'+str(n_bin_z)+'\\'

    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    try:
        with open(path+'neurons_concentration.pkl', 'rb') as f:
            [n_neu_in_sub_pyr,sub_net_pos_pyr,sub_net_pyr,n_neu_in_sub_int,sub_net_pos_int,sub_net_int] = pickle.load(f)
    except:
        variabilità_spaziale=np.zeros([3,2],int)

        variabilità_spaziale[0,0]=np.floor(min(posizioni_piramidali[:,1].min(),posizioni_interneuroni[:,1].min())).astype(int)
        variabilità_spaziale[1,0]=np.floor(min(posizioni_piramidali[:,2].min(),posizioni_interneuroni[:,2].min())).astype(int)
        variabilità_spaziale[2,0]=np.floor(min(posizioni_piramidali[:,3].min(),posizioni_interneuroni[:,3].min())).astype(int)


        variabilità_spaziale[0,1]=np.ceil(max(posizioni_piramidali[:,1].max(),posizioni_interneuroni[:,1].max())).astype(int)
        variabilità_spaziale[1,1]=np.ceil(max(posizioni_piramidali[:,2].max(),posizioni_interneuroni[:,2].max())).astype(int)
        variabilità_spaziale[2,1]=np.ceil(max(posizioni_piramidali[:,3].max(),posizioni_interneuroni[:,3].max())).astype(int)


        bin_x=int(np.ceil((variabilità_spaziale[0,1]-variabilità_spaziale[0,0])/n_bin_x))
        bin_y=int(np.ceil((variabilità_spaziale[1,1]-variabilità_spaziale[1,0])/n_bin_y))
        bin_z=int(np.ceil((variabilità_spaziale[2,1]-variabilità_spaziale[2,0])/n_bin_z))



        sub_net_pos_pyr=[]
        sub_net_pyr=[]
        n_neu_in_sub_pyr=[]
        sub_net_pos_int=[]
        sub_net_int=[]
        n_neu_in_sub_int=[]
        n_neu=0
        for i in range(variabilità_spaziale[0,0],variabilità_spaziale[0,1],bin_x):
            for j in range(variabilità_spaziale[1,0],variabilità_spaziale[1,1],bin_y):
                for k in range(variabilità_spaziale[2,0],variabilità_spaziale[2,1],bin_z):
                    print(min(i+bin_x,variabilità_spaziale[0,1]),min(j+bin_y,variabilità_spaziale[1,1]), min(k+bin_z,variabilità_spaziale[2,1]))
                    x_constraints=np.logical_and(posizioni_piramidali[:, 1] >= i, posizioni_piramidali[:, 1] < i + bin_x)
                    y_constraints = np.logical_and(posizioni_piramidali[:, 2] >= j, posizioni_piramidali[:, 2] < j + bin_y)
                    z_constraints = np.logical_and(posizioni_piramidali[:, 3] >= k, posizioni_piramidali[:, 3] < k + bin_z)
                    neu_in_sub_pyr=posizioni_piramidali[np.logical_and(np.logical_and(x_constraints,y_constraints),z_constraints),0]
                    if neu_in_sub_pyr.shape[0]>0:
                        sub_net_pyr.append(neu_in_sub_pyr)
                        sub_net_pos_pyr.append([i+bin_x/2,j+bin_y/2,k+bin_z/2])
                        n_neu_in_sub_pyr.append(sub_net_pyr[-1].shape[0])
                        n_neu=n_neu+sub_net_pyr[-1].shape[0]

                    x_constraints = np.logical_and(posizioni_interneuroni[:, 1] >= i,posizioni_interneuroni[:, 1] < i + bin_x)
                    y_constraints = np.logical_and(posizioni_interneuroni[:, 2] >= j,posizioni_interneuroni[:, 2] < j + bin_y)
                    z_constraints = np.logical_and(posizioni_interneuroni[:, 3] >= k,posizioni_interneuroni[:, 3] < k + bin_z)
                    neu_in_sub_int = posizioni_interneuroni[
                        np.logical_and(np.logical_and(x_constraints, y_constraints), z_constraints), 0]
                    if neu_in_sub_int.shape[0] > 0:
                        sub_net_int.append(neu_in_sub_int)
                        sub_net_pos_int.append([i + bin_x / 2, j + bin_y / 2, k + bin_z / 2])
                        n_neu_in_sub_int.append(sub_net_int[-1].shape[0])
                        n_neu = n_neu + sub_net_int[-1].shape[0]
        with open(path + 'neurons_concentration.pkl', 'wb') as f:
                pickle.dump([n_neu_in_sub_pyr, sub_net_pos_pyr, sub_net_pyr, n_neu_in_sub_int, sub_net_pos_int,
                 sub_net_int], f)

    return [n_neu_in_sub_pyr,sub_net_pos_pyr,sub_net_pyr,n_neu_in_sub_int,sub_net_pos_int,sub_net_int]