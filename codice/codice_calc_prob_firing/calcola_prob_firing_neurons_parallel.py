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

def calcola_pr_in_uscita3(id_pyr,fat_molt):
    print(id_pyr)
    Combinazioni_tot=0
    Combinazioni_pos=0
    for k in range (int(np.floor(n_con_tot_on_pyr[id_pyr]/10)),n_con_tot_on_pyr[id_pyr]):
        #quante sono le combinazioni in ingresso di k stimoli sul neurone id_pyr (per k>di un decimo del numero di connessioni totali (eccitatorie e inibitorie)
        Combinazioni_tot=Combinazioni_tot+math.comb(n_con_tot_on_pyr[id_pyr], k)
        #combinazioni in input con più stimoli da sinapsi eccitatorie che inibitorie
        for i in range(int(np.floor(k*fat_molt))+1,min(k,n_con_ex_on_pyr[id_pyr])):
            if (k - n_con_in_on_pyr[id_pyr] >= i) and i <= k:
                Combinazioni_pos = Combinazioni_pos + math.comb(n_con_ex_on_pyr[id_pyr], i)*math.comb(n_con_in_on_pyr[id_pyr], k-i)


    return  [Combinazioni_pos/Combinazioni_tot,Combinazioni_pos,Combinazioni_tot]


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
fat_molt=3/5;
path='C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python\\'
with open(path+'Pyr_connection_info.pkl', 'rb') as f:
    [id, n_con_ex_on_pyr, n_con_in_on_pyr] = pickle.load(f)

with open(path + 'Pyr_connection_info_out.pkl', 'rb') as f:
    [id, n_con_ex_on_pyr_out, n_con_in_on_pyr_out] = pickle.load(f)

try:
    with open(path + 'Firing_Probability_'+str(fat_molt)+'.pkl', 'rb') as f:
        [id,probabilità_firing, Combinazioni_tot, Combinazioni_pos] = pickle.load(f)
except:

    N_pyr=id.__len__()

    n_con_ex_on_pyr=n_con_ex_on_pyr.astype(int)
    n_con_in_on_pyr=n_con_in_on_pyr.astype(int)
    n_con_ex_on_pyr_out=n_con_ex_on_pyr_out.astype(int)
    n_con_in_on_pyr_out=n_con_in_on_pyr_out.astype(int)


    n_con_tot_on_pyr=n_con_ex_on_pyr+n_con_in_on_pyr
    n_con_tot_on_pyr_out=n_con_ex_on_pyr_out+n_con_in_on_pyr_out

    n_workers=os.cpu_count()


    Ltt = Parallel(n_jobs=n_workers, verbose=50)(delayed(calcola_pr_in_uscita3)(i,fat_molt) for i in range(N_pyr))


    results = np.array(Ltt)

    probabilità_firing=results[:,0]
    Combinazioni_pos =results[:,1]
    Combinazioni_tot =results[:,2]
    id=np.arange(1,N_pyr+1,1)
    with open(path+'Firing_Probability_'+str(fat_molt)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([id,probabilità_firing, Combinazioni_tot, Combinazioni_pos], f)

    counts, bins = np.histogram(probabilità_firing, 100)
    plt.stairs(counts, bins)
    plt.show()

    probabilità_firing.max()

    f_pos[pos_neuron_list[10]][np.where(probabilità_firing > probabilità_firing.max() / 2)[:], 1:4]
    pos_neuron_with_higher_prob=f_pos[pos_neuron_list[10]][np.where(probabilità_firing>probabilità_firing.max()/2)[0],1:4]

    points = go.Scatter3d(x=pos_neuron_with_higher_prob[:, 0],
                          y=pos_neuron_with_higher_prob[:, 1],
                          z=pos_neuron_with_higher_prob[:, 2],
                          mode='markers',  # 'lines+markers',
                          marker=dict(size=1,
                                      showscale=True),

                          )

    layout = go.Layout(title='num neurons with high probability of firing',
                       margin=dict(l=0,
                                   r=0,
                                   b=0,
                                   t=0)
                       )

    n_bin_x = 10
    n_bin_y = 20
    n_bin_z = 17
    path2 = 'C:\\Users\\emili\\Desktop\\CNR\\2023-24\\figure4ae_data_code\\codice_python\\results_bin_x_' + str(
        n_bin_x) + '_bin_y_' + str(n_bin_y) + '_bin_z_' + str(n_bin_z) + '\\'

    with open(path+'data_last.pkl', 'rb') as f:
        [n_bin_x,n_bin_y,n_bin_z,variabilità_spaziale,neurons_stim_beetween_classes,n_connessioni_beetween_sub_net,vicini,n_sub_net,sub_net_pos,sub_net,n_neu_in_sub] = pickle.load(f)


    fig = go.Figure(data=points, layout=layout)

    for i in range(sub_net_pos.__len__()):
        fig.add_trace(parallelepipedo(bin_x, bin_y, bin_z, sub_net_pos[i][0], sub_net_pos[i][1], sub_net_pos[i][2],
                                      'rgba(100,0,100,0.1)'))
