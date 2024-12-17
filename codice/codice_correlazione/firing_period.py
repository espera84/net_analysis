import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red
from gini_WB import compute_Theil_W_B,compute_Theil
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
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
from scipy.optimize import curve_fit

def calcola_fn(t):

    global spk_list,fn_tutti,ns_fn_tutti

    fn = spk_list[np.logical_and(iniz_intervals[t] < spk_list[:, 1], spk_list[:, 1] < fine_intervals[t]), 0]

    [aux, n_spk_aux] = np.unique(fn, return_counts=True)

    fn_tutti[t]=aux
    ns_fn_tutti[t]=n_spk_aux
    n_spk_for_neuron[aux.astype(int), t] = n_spk_for_neuron[aux.astype(int), t] + n_spk_aux

def calcola_spk_for_neuron(t):

    global spk_list,fn_tutti,ns_fn_tutti

    fn = spk_list[np.logical_and(iniz_intervals[t] < spk_list[:, 1], spk_list[:, 1] < fine_intervals[t]), 0]
    [aux, n_spk_aux] = np.unique(fn, return_counts=True)
    n_spk_for_neuron[aux.astype(int), t]=n_spk_for_neuron[aux.astype(int),t]+n_spk_aux


#def fsin(t,f,A,phi):
#    return A*np.sin(2*np.pi*f*t+phi)
def fsin(t,f,phi):
    return np.sin(2*np.pi*f*t+phi)

plt.ioff()
work_path= "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
sigla="sl9"
# work_path[1] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
# sigla[1]="ls5"
# work_path[2] = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
# sigla[2]="bkg_5hz"


save_neu_fig=False#True
n_neuroni=288027
setting_file = "./configuration.json"
sim_conf = json.load(open('%s' % (setting_file), 'r'))
t_initial_analysis = sim_conf['start_time']
t_final_analysis = sim_conf['end_time']  # 0#5000#
interval_dim = sim_conf['bins_dimension']
interval_dim=50
n_workers = -1
perc_attivi = sim_conf['percentual_of_firing_bins_for_active']
soglia_attivi = perc_attivi * (t_final_analysis - t_initial_analysis) / interval_dim
perc_corr = sim_conf['percentual_of_egual_bins_for_correlation']
soglia_di_correlazione = perc_corr * (t_final_analysis - t_initial_analysis) / interval_dim
n_shift = sim_conf['n_shift']

results_path="./interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
try:
    os.mkdir(results_path)
except FileExistsError:
    pass
#results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"

list_of_list_spk=h5py.File(work_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]
fn=np.unique(spk_list[:,0])


iniz_intervals=[]
fine_intervals=[]



t_iniz=t_initial_analysis
t_fin=t_initial_analysis+interval_dim

while(t_fin<=t_final_analysis):
    iniz_intervals.append(float(t_iniz))
    fine_intervals.append(float(t_fin))
    t_iniz=t_fin
    t_fin=t_fin+interval_dim
fine_intervals=np.array(fine_intervals)
iniz_intervals=np.array(iniz_intervals)
time=(iniz_intervals+fine_intervals)/2


filename_in = "connections_inh.hdf5"
filename_PC = "SP_PC_to_SP_PC.hdf5"
filename_pos="positions.hdf5"
#with  as f:
f_in=h5py.File(filename_in, "r")
in_connection_list=list(f_in.keys())

f_pyr=h5py.File(filename_PC, "r")
pyr_connection_list=list(f_pyr.keys())

N_pyr=f_pyr[pyr_connection_list[0]][:,0].max()

f_pos=h5py.File(filename_pos, "r")
pos_neuron_list=list(f_pos.keys())

n_spk_for_neuron=np.zeros((n_neuroni,time.shape[0]),dtype=int)

fn_tutti = np.empty((time.shape[0]), dtype=object)
ns_fn_tutti = np.empty((time.shape[0]), dtype=object)
is_spking = np.empty((n_neuroni, time.shape[0]), dtype=bool)
stdev=np.zeros((n_neuroni),dtype=int)
fr=np.zeros((n_neuroni),dtype=float)
rad=np.zeros((n_neuroni),dtype=float)
std_fr=np.zeros((n_neuroni),dtype=float)
periods=np.zeros((n_neuroni),dtype=float)
dist_tot_norm=np.ones((n_neuroni),dtype=float)*10000

try:
    with open('firing_info_'+sigla+'_'+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+'.pkl', 'rb') as f:
        [fn_tutti,ns_fn_tutti,n_spk_for_neuron] = pickle.load(f)
except:
    calcola_fn(0)

    Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_fn)(t) for t in range(1, time.shape[0]))
    with open('firing_info_'+sigla+'_'+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([fn_tutti,ns_fn_tutti,n_spk_for_neuron], f)

# for i in range(1000):
#     stdev[i]=np.std(n_spk_for_neuron[i,:])
#     if stdev[i]>1:
#         print(i)
#         #plt.figure(i)
#         #plt.plot(time,n_spk_for_neuron[i,:])
#         #plt.show()



######
for id_neu in range(1000,2000):#n_neuroni):
    stdev[id_neu] = np.std(n_spk_for_neuron[id_neu, :])
    if stdev[id_neu]>0.2:
        hist=n_spk_for_neuron[id_neu,:]
        bins=np.concatenate((iniz_intervals,[t_final_analysis]))

        soglia =hist.max()/2#2# (hist.mean() + hist.min()) * 2 / 3
        hist_sup_soglia = hist >= soglia

        pos_init_intervals =np.where(np.logical_and(hist_sup_soglia[1:], np.logical_not(hist_sup_soglia[:-1])))[0] + 1
        pos_init_intervals=np.where((hist[1:]-hist[:-1])>soglia)
        init_intervals = iniz_intervals[pos_init_intervals]

        # if hist_sup_soglia[-1]==True:
        #     pos_end_intervals=np.concatenate((np.where(np.logical_and(hist_sup_soglia[:-1], np.logical_not(hist_sup_soglia[1:])))[0],np.array([hist_sup_soglia.__len__()-1])))
        #
        # else:
        #     pos_end_intervals=np.where(np.logical_and(hist_sup_soglia[:-1],np.logical_not(hist_sup_soglia[1:])))[0]
        # end_intervals = fine_intervals[pos_end_intervals]

        # pos_hist_sup_sog = np.where(hist_sup_soglia == True)
        # vet_aux = pos_hist_sup_sog[0][:-1] == pos_hist_sup_sog[0][1:] - 1
        # in_interval = False
        # first_element = False
        # init_intervals = []
        # end_intervals = []
        # pos_init_intervals = []
        # pos_end_intervals = []
        # for i in range(vet_aux.__len__()):
        #     if not in_interval:
        #         if vet_aux[i] == True:
        #             in_interval = True
        #             first_element = True
        #             init_intervals.append(iniz_intervals[pos_hist_sup_sog[0][i]])#bins[pos_hist_sup_sog[0][i]])
        #             end_intervals.append(fine_intervals[pos_hist_sup_sog[0][i]])#bins[pos_hist_sup_sog[0][i]])
        #             pos_init_intervals.append(pos_hist_sup_sog[0][i])
        #             pos_end_intervals.append(pos_hist_sup_sog[0][i])
        #     else:
        #         if vet_aux[i] == True:
        #             if vet_aux[i - 1] == True:
        #                 end_intervals[-1] = fine_intervals[pos_hist_sup_sog[0][i + 1]]#bins[pos_hist_sup_sog[0][i + 1]]
        #                 pos_end_intervals[-1] = pos_hist_sup_sog[0][i + 1]
        #         else:
        #             in_interval = False

        if init_intervals.__len__()>1:
            #time_to_fit=init_intervals
            time_to_fit=time[pos_init_intervals]
            periods[id_neu]=(np.mean(time_to_fit[1:]-time_to_fit[:-1])/1000)
            fr[id_neu]=1/periods[id_neu]
            std_fr[id_neu]=np.std(time_to_fit[1:]-time_to_fit[:-1])/1000

            dt=np.zeros(time_to_fit.__len__())
            dist=np.zeros((time_to_fit.__len__(),np.arange(-(1000 / fr[id_neu]),(1000 / fr[id_neu])).__len__()))
            for i in range(time_to_fit.__len__()):
                print( t_initial_analysis + (1000 / fr[id_neu]) * i," ", time_to_fit[i])
                dt[i] = time_to_fit[i]- (t_initial_analysis + (1000 / fr[id_neu]) * i)
                dist[i,:]=dt[i]-np.arange(-(1000 / fr[id_neu]),(1000 / fr[id_neu]))
            dist_tot=np.absolute(dist).sum(0)
            dist_tot_norm[id_neu] = (dist_tot.min()/(1000 / fr[id_neu]))/dist.__len__()
            rad[id_neu]=np.arange(-(1000 / fr[id_neu]),(1000 / fr[id_neu]))[dist_tot.argmin()]/(1000 / fr[id_neu])*2*np.pi#*360
            pars0=[fr[id_neu],np.pi/2-rad[id_neu]]

            if save_neu_fig:
                plt.figure(id_neu)


                plt.plot(time,fsin((time-t_initial_analysis)/1000,*pars0),'--r', label='funzione di fit con parametri ottimizzati')

                plt.scatter(time_to_fit,np.ones(time_to_fit.__len__()))


                plt.plot(time,hist)
                plt.savefig(results_path+"fit_neu "+str(id_neu)+"_"+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+"_"+str(soglia)+"_"+sigla+".png")
            #plt.close(id_neu)
#plt.figure()
#plt.hist(fr[fr>0],30)
#plt.savefig("hist_fr_"+sigla+'_'+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+".png")
plt.figure()
counts, bins = np.histogram(fr[fr>0],30)
plt.stairs(counts, bins)
plt.savefig(results_path+"hist_fr_"+sigla+'_'+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+"_"+str(soglia)+".png")

plt.figure()
plt.scatter(fr[fr>0],rad[fr>0])
plt.savefig(results_path+"scatter_fr_rad_"+sigla+'_'+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+"_"+str(soglia)+".png")


posizioni_neuroni=f_pos[pos_neuron_list[0]][:]
print(str(f_pos[pos_neuron_list[0]][:].__len__())+" "+str(posizioni_neuroni.__len__()))
for j in range(1,len(pos_neuron_list)):
    print(str(f_pos[pos_neuron_list[i]][:].__len__())+" "+str(posizioni_neuroni.__len__()))
    posizioni_neuroni = np.concatenate((posizioni_neuroni, f_pos[pos_neuron_list[j]][:]))



camera=dict(eye=dict(x=2, y=0, z=0))
points = go.Scatter3d(x=posizioni_neuroni[0::10, 1],
                      y=posizioni_neuroni[0::10, 2],
                      z=posizioni_neuroni[0::10, 3],
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
color_discrete_sequence = ["orange", "red", "green", "blue", "pink"]
for i in range(1,bins.__len__()):
    neuron_to_plt=np.in1d(posizioni_neuroni[:,0],np.where(np.logical_and(fr>bins[i-1],fr<bins[i])))
    fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                        y=posizioni_neuroni[neuron_to_plt,2],
                        z=posizioni_neuroni[neuron_to_plt,3],
                        mode='markers',
                        name='frequency_map_' + sigla,
                        marker=dict(size=5,
                                    #colorscale='pinkyl',
                                    color=i,#color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                           #colorscale='Plotly3',
                                           #showscale=False,
                                    )
                        )
fig3.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

fig3.write_html(results_path+"frequency_map_"+sigla+"_"+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+"_"+str(soglia)+".html")


fig3 = go.Figure(data=points, layout=layout)
color_discrete_sequence = ["orange", "red", "green", "blue", "pink"]
for i in range(1,bins.__len__()):
    neuron_to_plt=np.in1d(posizioni_neuroni[:,0],np.where(np.logical_and(fr>bins[i-1],fr<bins[i])))
    fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                        y=posizioni_neuroni[neuron_to_plt,2],
                        z=posizioni_neuroni[neuron_to_plt,3],
                        mode='markers',
                        name='frequency ' + str(bins[i-1])+str(bins[i]),
                        marker=dict(size=5,
                                    #colorscale='pinkyl',
                                    color=rad[np.where(np.logical_and(fr>bins[i-1],fr<bins[i]))],#color_discrete_sequence[i],#px.colors.qualitative.Antique[i],#l,#list(colori),#px.colors.qualitative.Antique[i],#                                       color=px.colors.qualitative.Plotly[i],#float(i / n_comp_connesse),
                                           #colorscale='Plotly3',
                                           #showscale=False,
                                    )
                        )
fig3.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

fig3.write_html(results_path+"frequency_map_with_phase_color_"+sigla+"_"+str(t_initial_analysis)+'_'+str(t_final_analysis)+'_'+str(interval_dim)+"_"+str(soglia)+".html")
#plt.ion()
plt.figure()
plt.scatter(np.arange(dist_tot_norm[dist_tot_norm<10000].__len__()),dist_tot_norm[dist_tot_norm<10000])
#plt.show()