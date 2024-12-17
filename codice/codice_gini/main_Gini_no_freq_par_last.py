import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red
from gini_WB import compute_gini_WB_without_freq_parallel
import matplotlib.pyplot as plt
import pickle
import os
import csv
import pandas as pd

import h5py
from utility import unify_families
import time


import json

t1 = time.time()
setting_file="./sim_configuration.json"
sim_conf = json.load(open('%s'%(setting_file), 'r'))
path = sim_conf['Path']
start_time = sim_conf['Time_setting']['Start_time']
simulation_time= sim_conf['Time_setting']['simulation_time']
bin_size=sim_conf['Time_setting']['temporal_bin_size']
n_pezzi=sim_conf['n_pezzi']
path_results=path+"results/"#'neu_families'+'-'.join(labels)+"/"
path_results_gini=path_results+"Gini_results/"


filename_pos="positions.hdf5"
f_pos=h5py.File(filename_pos, "r")
neurons_families_list=list(f_pos.keys())
labels=neurons_families_list
n_pop=neurons_families_list.__len__()

try:
    os.mkdir(path_results)
except FileExistsError:
    pass



try:
    os.mkdir(path_results_gini)
except FileExistsError:
    pass


N_SP_PC=f_pos['SP_PC'][:,0].shape[0]
N_AA=f_pos['AA'][:,0].shape[0]
N_BP=f_pos['BP'][:,0].shape[0]
N_BS=f_pos['BS'][:,0].shape[0]
N_CCKBC=f_pos['CCKBC'][:,0].shape[0]
N_IVY=f_pos['IVY'][:,0].shape[0]
N_NGF=f_pos['NGF'][:,0].shape[0]
N_OLM=f_pos['OLM'][:,0].shape[0]
N_PPA=f_pos['PPA'][:,0].shape[0]
N_PVBC=f_pos['PVBC'][:,0].shape[0]
N_SCA=f_pos['SCA'][:,0].shape[0]
N_TRI=f_pos['TRI'][:,0].shape[0]

tot_neu=0
neu_ind_interval=[]
norm_fact=np.zeros(n_pop)
for i in range(n_pop):
    neu_ind_interval.append([int(f_pos[neurons_families_list[i]][:, 0].min()), int(f_pos[neurons_families_list[i]][:, 0].max())])
    norm_fact[i]=np.log(f_pos[neurons_families_list[i]][:,0].shape[0])
    tot_neu=tot_neu+f_pos[neurons_families_list[i]][:,0].shape[0]
norm_fact_tot=np.log(tot_neu)
#list_of_list_spk=[]
#list_of_list_spk = pd.read_csv(path+activity.csv",header=None)
#spk_list=np.array(list_of_list_spk)

list_of_list_spk = h5py.File(path + "activity_network.hdf5", "r")
spk_list = list_of_list_spk['spikes'].astype('float64')[:]
try:
    with open(path_results_gini+'GT_features.pkl', 'rb') as f:
        [ns_at_time,ns_tot_at_time,ns_pop_at_time, mu,mu_IC] = pickle.load(f)
except:

    [ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC]=features_extraction_different_format_red(spk_list,n_pop,start_time,simulation_time,bin_size,neu_ind_interval)
    with open(path_results_gini+'GT_features.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC], f)

try:
    with open(path_results_gini+'Gini_Analisys_data.pkl', 'rb') as f:
        [n_pop, labels, start_time, simulation_time, bin_size, ns_at_time, ns_tot_at_time, ns_pop_at_time, mu, mu_IC,
         Gini_JH, Gini_JJ, Gini_B, Gini_W, Gini_B_tot, Gini_Cumulative, Gini_JJ_interval, Gini_W_interval,
         Gini_B_interval, Gini_B_interval_tot, Gini_interval] = pickle.load(f)
except:

    [Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini_Cumulative,Gini_JJ_interval,Gini_W_interval,Gini_B_interval,Gini_B_interval_tot,Gini_interval]=compute_gini_WB_without_freq_parallel(ns_at_time,ns_tot_at_time,ns_pop_at_time, mu,mu_IC,n_pezzi)


with open(path_results_gini+'Gini_Analisys_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([n_pop, labels, start_time,simulation_time,bin_size,ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC,Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini_Cumulative,Gini_JJ_interval,Gini_W_interval,Gini_B_interval,Gini_B_interval_tot,Gini_interval], f)


num_interval=len(Gini_B_tot)

for i in range(n_pop):
    plt.figure();
    plt.plot(np.array(range(num_interval-1)) * bin_size + start_time, Gini_JJ[i][1:], label = labels[i]  )
    plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time (ms)')
    plt.ylabel('Gini index cumulative ')
    plt.savefig(path_results_gini+'Gini_index_cumulative_'+labels[i]+".png")

Gini_JJ_for_interval=np.zeros([n_pop,2*(ns_tot_at_time.size)])

for pop in range(n_pop):
    for t in range(ns_tot_at_time.size):
        Gini_JJ_for_interval[pop,2*t]=Gini_JJ_interval[pop,t]
        Gini_JJ_for_interval[pop, 2 * t+1]=Gini_JJ_interval[pop,t]

for i in range(n_pop):
    plt.figure();
    plt.plot(np.ceil(np.array(range(2*(len(Gini_B_tot)-1)))/2)*bin_size+start_time,Gini_JJ_for_interval[i][2:], label=labels[i]  )
    plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time (ms)')
    plt.ylabel('Gini index interval ')
    plt.savefig(path_results_gini + 'Gini_index_for_inteval_' + labels[i] + ".png")







for i in range(n_pop):
    plt.figure();
    plt.plot(np.array(range(num_interval-1))*bin_size+start_time,Gini_W[i][1:], label=labels[i]  )
    plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time (ms)')
    plt.ylabel('within-subsystem Gini index ')
    plt.savefig(path_results_gini + "Gini_Cumulative_within_"+labels[i]+".png")
    plt.figure();


plt.plot(np.array(range(len(Gini_W[i])))*bin_size+start_time,Gini_B_tot, label='gini_B_tot'  )
plt.plot(np.array(range(len(Gini_W[i])))*bin_size+start_time,np.nansum(Gini_W,axis=0), label='gini_W_tot'  )
plt.plot(np.array(range(len(Gini_W[i])))*bin_size+start_time,Gini_Cumulative, label='gini'  )
plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
plt.xlabel('Time (ms)')
plt.ylabel('gini index')
#plt.savefig(path + "Gini_Cumulative_within_between.svg")

plt.savefig(path_results_gini + "Gini_Cumulative_wb.png")
#plt.ylabel('between-subsystem Gini index ')



t2=time.time()-t1
