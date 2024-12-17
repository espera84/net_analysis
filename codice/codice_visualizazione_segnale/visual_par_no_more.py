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


def calcola_fn(t):

    global fir_neu_pos_tutti,center_weighted,center,posizioni
    global spk_list,fn_tutti,ns_fn_tutti,n_of_classes,iniz_intervals,fine_intervals,n_conn_out_on_ex_tot,n_conn_out_on_in_tot
    global n_sp,n_neu_conn_out_on_ex_tot,n_neu_conn_out_on_in_tot,n_con_pyr_on_in,n_con_pyr_on_ex,n_sp_neu, n_sp_X_neu


    firing_neuron_pos=np.empty((0,5), dtype=float)

    fn = spk_list[np.logical_and(iniz_intervals[t] < spk_list[:, 1], spk_list[:, 1] <= fine_intervals[t]), 0]

    n_conn_out_on_ex_tot[t] = n_con_pyr_on_ex[np.ndarray.tolist(fn[fn[:] <= n_pyr].astype(int) - 1)].sum()
    n_conn_out_on_in_tot[t] = n_con_pyr_on_in[np.ndarray.tolist(fn[fn[:] <= n_pyr].astype(int) - 1)].sum()
    n_sp[t]=np.shape(fn)[0]
    [aux, n_spk_aux] = np.unique(fn, return_counts=True)
    #print("aggiorno fn tutti ",t)
    fn_tutti[t]=aux
    ns_fn_tutti[t]=n_spk_aux

    n_neu_conn_out_on_ex_tot[t] = n_con_pyr_on_ex[np.ndarray.tolist(aux[aux[:] <= n_pyr].astype(int) - 1)].sum()
    n_neu_conn_out_on_in_tot[t] =n_con_pyr_on_in[np.ndarray.tolist(aux[aux[:] <= n_pyr].astype(int) - 1)].sum()

    n_sp_neu[t]=np.shape(aux)[0]
    n_sp_X_neu[t]=n_sp[t] / n_sp_neu[t]


    for i in range(n_of_classes):
        #print(t, i)
        # firing_neuron_pos.append([f_pos[pos_neuron_list[i]][np.isin(f_pos[pos_neuron_list[i]][:,0],aux),1:],time[t]]) #non consideriamo il numero di occorrenze
        pos_fn = posizioni[i][
                 np.isin(posizioni[i][:, 0], aux, assume_unique=True, kind='sort'), :]
        n_spk_fn = n_spk_aux[np.isin(aux, posizioni[i][
            np.isin(posizioni[i][:, 0], aux, assume_unique=True, kind='sort'), 0])]
        ####ATTENZIONE la corrispondenza tra n_spk_fn e pos_fn dipende dal fatto che sia aux sia f_pos[pos_neuron_list[i]][:] sono ordinati in funzione dell'id dei neuroni
        ####se f_pos[pos_neuron_list[i]][:] non fosse ordinato in funzione dell'id del neurone dovrebbe essere sostituito con f_pos[pos_neuron_list[i]][np.argsort(f_pos[pos_neuron_list[i]][:,0]),:] all'interno della definizione di pos_fn



        firing_neuron_pos=np.concatenate((firing_neuron_pos,np.concatenate((pos_fn, n_spk_fn.reshape([n_spk_fn.__len__(), 1])),
                           axis=1)))
        #print(t, i)
        #print(firing_neuron_pos[i].__len__())

    fir_neu_pos_tutti[t]=firing_neuron_pos


    #center[t]=np.append(fir_neu_pos_tutti[fir_neu_pos_tutti[:, 4] == t, 1:4].mean(axis=0), time[t])
    print(t)
    #center_weighted[t]=np.append(np.average(fir_neu_pos_tutti[fir_neu_pos_tutti[:, 4] == t, 1:4], axis=0,weights=fir_neu_pos_tutti[fir_neu_pos_tutti[:, 4] == t, 5]), time[t])


def calcola_fn_no_more(t):
    global fir_neu_pos_tutti_no_more,posizioni,n_of_classes,fn_tutti

    firing_neuron_no_more_pos =np.empty((0,4), dtype=float)

    fn_nm=np.isin(fn_tutti[t-1],fn_tutti[t], assume_unique=True, invert=True, kind='sort')
    aux_nm = fn_tutti[t-1][fn_nm]

    for i in range(n_of_classes):
        print(t, i)

        pos_fn_nm = posizioni[i][np.isin(posizioni[i][:, 0], aux_nm, assume_unique=True, kind='sort'), :]

        firing_neuron_no_more_pos = np.concatenate((firing_neuron_no_more_pos, pos_fn_nm))

    fir_neu_pos_tutti_no_more[t] = firing_neuron_no_more_pos


def calcola_fn_dif(t):
    global fir_neu_pos_tutti_diff,posizioni,n_sp_dif,n_sp_neu_dif,n_sp_X_neu_dif,n_of_classes
    global spk_list, fn_tutti, ns_fn_tutti

    firing_neuron_pos_diff =np.empty((0,5), dtype=float)

    fn_dif=np.isin(fn_tutti[t],fn_tutti[t-1], assume_unique=True, invert=True, kind='sort')
    aux_diff = fn_tutti[t][fn_dif]
    n_spk_aux_diff = ns_fn_tutti[t][fn_dif]    #NUM SPIKES NEURONI DIF
    n_sp_dif[t]=n_spk_aux_diff.sum()
    n_sp_neu_dif[t]=np.shape(aux_diff)[0]
    n_sp_X_neu_dif[t]=n_sp_dif[t] / n_sp_neu_dif[t]

    for i in range(n_of_classes):
        print(t, i)

        pos_fn_diff = posizioni[i][np.isin(posizioni[i][:, 0], aux_diff, assume_unique=True, kind='sort'), :]
        n_spk_fn_diff = n_spk_aux_diff[np.isin(aux_diff, posizioni[i][np.isin(posizioni[i][:, 0], aux_diff, assume_unique=True, kind='sort'), 0])]

        firing_neuron_pos_diff = np.concatenate((firing_neuron_pos_diff, np.concatenate((pos_fn_diff, n_spk_fn_diff.reshape([n_spk_fn_diff.__len__(), 1])),axis=1)))

    fir_neu_pos_tutti_diff[t] = firing_neuron_pos_diff

def find_stimulation_neurons3(ind):

    global clusters_diff, in_conn
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


def find_neurons_stimulated(ind):

    global clusters_no_more, out_conn,n_pyr
    lung = clusters_no_more[ind].__len__()
    if lung != 0:
        n_cl = int(clusters_no_more[ind][:, 4].max()) + 1
        id_neu_stimolati = np.empty((n_cl,), dtype=object)

        for i in range(n_cl):
            ind_neu_cluster =clusters_no_more[ind][clusters_no_more[ind][:, 4] == i,0].astype(int)
            ind_neu_cluster=ind_neu_cluster[ind_neu_cluster<n_pyr]
            out = out_conn[ind_neu_cluster]
            out2=[]
            for k in range(out.__len__()):
                #print(k)
                out2 = np.unique(out2 + out[k]).tolist()
            id_neu_stimolati[i] = np.unique(out2)


        return id_neu_stimolati
    else:
        return np.empty(0)


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

posizioni=[]
for i in  range(len(pos_neuron_list)):
    posizioni.append(f_pos[pos_neuron_list[i]][:])


add_struct_cicle=False
save_all_images=True
threshold_connection=850


n_bins=500

t_initial_analysis=1500
t_final_analysis=2000
interval_dim=50

m_sam=100
eps=150

n_workers = -1

work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_dal_20_6/test_num_nodes_simdata_mouse_1c430bc3-cfd2-45b9-8dc6-00ab32edb083/results/"


#list_of_list_spk = pd.read_csv(work_path+"activity.csv", header=None)
#spk_list=np.array(list_of_list_spk)

list_of_list_spk=h5py.File(work_path+"activity_network.hdf5", "r")
spk_list=list_of_list_spk['spikes'].astype('float64')[:]


counts, bins = np.histogram(spk_list[:,1],n_bins)
plt.stairs(counts, bins)
#plt.show(block=True)




bin_size=bins[1]-bins[0]
value=""
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
plt.stairs(counts, bins)

path=work_path+'T_A_C_int_from_'+str(t_initial_analysis)+'_to_'+str(t_final_analysis)+'_'+str(interval_dim)+'/'
path_cluster=path+'cl_'+str(m_sam)+'_'+str(eps)+'/'
path_sig=path+'all_signal_trasmission/'
path_fn=path+'firing_neurons_plot/'

try:
    os.mkdir(path)
except FileExistsError:
    pass
try:
    os.mkdir(path_fn)
except FileExistsError:
    pass


try:
    os.mkdir(path_cluster)
except FileExistsError:
    pass

try:
    os.mkdir(path_sig)
except FileExistsError:
    pass


plt.savefig(path+"firing_rate.png")

n_pyr=posizioni[10].__len__()

fir_neu_pos_tutti=np.empty((time.shape[0]), dtype=object)
fir_neu_pos_tutti_diff =np.empty((time.shape[0]), dtype=object)
fir_neu_pos_tutti_no_more=np.empty((time.shape[0]), dtype=object)

n_of_classes=pos_neuron_list.__len__()
n_sp=np.zeros(time.shape[0])
n_sp_dif=np.zeros(time.shape[0])
n_sp_neu=np.zeros(time.shape[0])
n_sp_X_neu=np.zeros(time.shape[0])
n_sp_X_neu_dif=np.zeros(time.shape[0])
n_sp_neu_dif=np.zeros(time.shape[0])
fn_tutti=np.empty((time.shape[0]), dtype=object)
ns_fn_tutti=np.empty((time.shape[0]), dtype=object)
center_diff=np.zeros((time.shape[0],4), dtype=float)
center_weighted_diff=np.zeros((time.shape[0],4), dtype=float)
center=np.zeros((time.shape[0],4), dtype=float)
center_weighted=np.zeros((time.shape[0],4), dtype=float)
n_conn_out_on_ex=[]
n_conn_out_on_in=[]
n_neu_conn_out_on_ex=[]
n_neu_conn_out_on_in=[]
n_intervalli=time.__len__()



fir_neu_pos_tutti_diff[0]=np.empty((0,5), dtype=float)
fir_neu_pos_tutti_no_more[0]=np.empty((0,4), dtype=float)

with open('Pyr_connection_info.pkl', 'rb') as g:
    [id,n_con_ex_on_pyr,n_con_in_on_pyr] = pickle.load(g)


with open('Pyr_connection_info_out.pkl', 'rb') as g:
    [id2,n_con_pyr_on_ex,n_con_pyr_on_in] = pickle.load(g)
try:
    with open(path + 'data.pkl', 'rb') as f:
        [fir_neu_pos_tutti_no_more,fir_neu_pos_tutti,fir_neu_pos_tutti_diff,time,n_sp_neu,n_sp,n_sp_X_neu,n_sp_neu_dif,n_sp_X_neu_dif,center_weighted_diff,center_weighted,center_diff,center,n_neu_conn_out_on_ex_tot,n_conn_out_on_ex_tot,n_neu_conn_out_on_in_tot,n_conn_out_on_in_tot] = pickle.load(f)
except:
    n_neu_conn_out_on_ex_tot=np.zeros(time.shape[0])
    n_neu_conn_out_on_in_tot = np.zeros(time.shape[0])
    n_conn_out_on_ex_tot=np.zeros(time.shape[0])
    n_conn_out_on_in_tot = np.zeros(time.shape[0])



    Parallel(n_jobs=n_workers, verbose=50,require='sharedmem')(delayed(calcola_fn)(t) for t in range(0,time.shape[0]))

    Parallel(n_jobs=n_workers, verbose=50,require='sharedmem')(delayed(calcola_fn_dif)(t) for t in range(1,time.shape[0]))

    Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_fn_no_more)(t) for t in range(1, time.shape[0]))

    for t in range(time.shape[0]):

        center[t]=np.append(fir_neu_pos_tutti[t][:,1:4].mean(axis=0), time[t])
        if t>0:
            center_diff[t] = np.append(fir_neu_pos_tutti_diff[t][:,1:4].mean(axis=0),time[t])

        if fir_neu_pos_tutti[t].__len__() > 0:
            center_weighted[t]=np.append(np.average(fir_neu_pos_tutti[t][:,1:4], axis=0,weights = fir_neu_pos_tutti[t][:,4]), time[t])
            if t > 0:
                center_weighted_diff[t]=np.append(np.average(fir_neu_pos_tutti_diff[t][:,1:4], axis=0,weights=fir_neu_pos_tutti_diff[t][:,4]),time[t])
        else:
            center_weighted[t] =center[t]
            if t>0:
                center_weighted_diff[t] = center_diff[t]



    with open(path +  'data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([fir_neu_pos_tutti_no_more,fir_neu_pos_tutti,fir_neu_pos_tutti_diff,time,n_sp_neu,n_sp,n_sp_X_neu,n_sp_neu_dif,n_sp_X_neu_dif,center_weighted_diff,center_weighted,center_diff,center,n_neu_conn_out_on_ex_tot,n_conn_out_on_ex_tot,n_neu_conn_out_on_in_tot,n_conn_out_on_in_tot], f)

print("data_loaded")
    #if t==0:
if save_all_images:
    print("saving_images")
    '''
    fig = px.scatter_3d(fir_neu_pos_tutti, x=1, y=2, z=3,color=4,opacity=0.1)
    fig.update_traces(marker_size = 1)
    #fig.show()
    fig.write_html(path +"scatter.html")


    fig = px.scatter_3d(fir_neu_pos_tutti_diff, x=1, y=2, z=3,color=4,opacity=0.5)
    fig.update_traces(marker_size = 1)
    #fig.show()
    fig.write_html(path +"scatter_differenze.html")
    '''
    #fig = px.scatter_3d(np.array([fir_neu_pos_tutti[1],fir_neu_pos_tutti_diff[1]]), x=0, y=1, z=2,color=3,opacity=0.5)
    #fig.update_traces(marker_size = 1)
    #fig.show()
    #fig.write_html(path +"scatter_differenze_picchi.html")



    print("global_scatter_saved")

    plt.figure()
    plt.scatter(time,n_sp_X_neu )
    plt.title("Spikes number")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean spikes number of neurons")
    #plt.show()
    plt.savefig(path +"number_of_spikes_for_neuron.png")


    plt.figure()
    plt.scatter(time[1:],n_sp_X_neu_dif[1:])
    plt.title("Spikes number")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean spikes number for each new neurons spiking")
    #plt.show()
    plt.savefig(path +"number_of_spikes_for_neuron_dif.png")



    intervals_len=fine_intervals-iniz_intervals

    plt.figure()
    plt.scatter(time,n_sp_X_neu/intervals_len )
    plt.title("Spikes number")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean frequency of neurons")
    #plt.show()
    plt.savefig(path +"freq_of_neuron.png")


    plt.figure()
    plt.scatter(time[1:],n_sp_X_neu_dif[1:]/intervals_len[1:] )
    plt.title("Spikes number")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean frequency of each new neuron spiking")
    #plt.show()
    plt.savefig(path +"freq_of_neuron_dif.png")






    plt.figure()
    plt.scatter(time[1:],n_sp_neu_dif[1:] )
    plt.title("number of new spiking neurons")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of new spiking neurons compared to the previously instant analized")
    #plt.show()
    plt.savefig(path +"number_of_spiking neurons_diff.png")


    plt.figure()
    plt.scatter(time,n_sp_neu )
    plt.title("spiking neurons")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of spiking neurons")
    #plt.show()
    plt.savefig(path +"number_of_spiking neurons.png")

    print("stat_info_saved")

    fig = px.scatter_3d(center_diff, x=0, y=1, z=2,color=3,opacity=1,title="geomentric center of new firing neurons compared to the previously instant analized")
    #fig.show()
    fig.write_html(path +"geometric_center_new_firing_neurons.html")




    plt.figure()
    plt.scatter(time,n_neu_conn_out_on_ex_tot )
    plt.title("number of excitatory neurons stimulated")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of excitatory neurons stimulated")
    #plt.show()
    plt.savefig(path +"number_of_excitatory_neurons_stimulated.png")


    plt.figure()
    plt.scatter(time,n_neu_conn_out_on_ex_tot )
    plt.title("number of stimulations to excitatory neurons")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of stimulations to excitatory neurons")
    #plt.show()
    plt.savefig(path +"number_of_stimulations_to_excitatory_neurons.png")


    plt.figure()
    plt.scatter(time,n_neu_conn_out_on_in_tot )
    plt.title("number of inhibitory neurons stimulated")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of inhibitory neurons stimulated")
    #plt.show()
    plt.savefig(path +"number_of_inhibitory_neurons_stimulated.png")


    plt.figure()
    plt.scatter(time,n_neu_conn_out_on_in_tot )
    plt.title("number of stimulations to inhibitory neurons")
    plt.xlabel("Time (ms)")
    plt.ylabel("number of stimulations to inhibitory neurons")
    #plt.show()
    plt.savefig(path +"number_of_stimulations_to_inhibitory_neurons.png")

    points = go.Scatter3d( x = np.array(center_diff)[:,0],
                           y = np.array(center_diff)[:,1],
                           z = np.array(center_diff)[:,2],
                           mode = 'lines+markers',
                           marker = dict( size = 5,
                                          color = np.array(center_diff)[:,3],
                                          showscale=True)
                         )


    layout = go.Layout(margin = dict( l = 0,
                                      r = 0,
                                      b = 0,
                                      t = 0)
                      )
    fig = go.Figure(data=points,layout=layout)



    fig.add_scatter3d(x=posizioni[10][0::10, 1],
                    y=posizioni[10][0::10, 2],
                    z=posizioni[10][0::10, 3],
                    mode='markers',
                    marker=dict(size=1,
                             color='gray',
                             showscale=True, opacity=0.3)
                 ,

                     )
    #fig.show()
    fig.write_html(path +"geometric_center_new_firing_neurons_linked.html")


    points = go.Scatter3d( x = np.array(center)[:,0],
                           y = np.array(center)[:,1],
                           z = np.array(center)[:,2],
                           mode = 'lines+markers',
                           marker = dict( size = 5,
                                          color = np.array(center)[:,3],
                                          showscale=True)
                         )


    layout = go.Layout(margin = dict( l = 0,
                                      r = 0,
                                      b = 0,
                                      t = 0)
                      )
    fig = go.Figure(data=points,layout=layout)


    fig.add_scatter3d(x=posizioni[10][0::10, 1],
                    y=posizioni[10][0::10, 2],
                    z=posizioni[10][0::10, 3],
                    mode='markers',
                    marker=dict(size=1,
                             color='gray',
                             showscale=True, opacity=0.3)
                 ,

                     )

    #fig.show()
    fig.write_html(path +"geometric_center_all_firing_neurons_linked.html")



    points = go.Scatter3d( x = np.array(center_weighted_diff)[:,0],
                           y = np.array(center_weighted_diff)[:,1],
                           z = np.array(center_weighted_diff)[:,2],
                           mode = 'lines+markers',
                           marker = dict( size = 5,
                                          color = np.array(center_weighted_diff)[:,3],
                                          showscale=True)
                         )


    layout = go.Layout(margin = dict( l = 0,
                                      r = 0,
                                      b = 0,
                                      t = 0)
                      )
    fig = go.Figure(data=points,layout=layout)


    fig.add_scatter3d(x=posizioni[10][0::10, 1],
                    y=posizioni[10][0::10, 2],
                    z=posizioni[10][0::10, 3],
                    mode='markers',
                    marker=dict(size=1,
                             color='gray',
                             showscale=True, opacity=0.3)
                 ,

                     )
    #fig.show()
    fig.write_html(path +"geometric_center_weighted_new_firing_neurons_linked.html")


    points = go.Scatter3d( x = np.array(center_weighted)[:,0],
                           y = np.array(center_weighted)[:,1],
                           z = np.array(center_weighted)[:,2],
                           mode = 'lines+markers',
                           marker = dict( size = 5,
                                          color = np.array(center_weighted)[:,3],
                                          showscale=True)
                         )


    layout = go.Layout(margin = dict( l = 0,
                                      r = 0,
                                      b = 0,
                                      t = 0)
                      )
    fig = go.Figure(data=points,layout=layout)

    fig.add_scatter3d(x=posizioni[10][0::10, 1],
                    y=posizioni[10][0::10, 2],
                    z=posizioni[10][0::10, 3],
                    mode='markers',
                    marker=dict(size=1,
                             color='gray',
                             showscale=True, opacity=0.3)
                 ,

                     )
    #fig.show()
    fig.write_html(path +"geometric_center_weighted_all_firing_neurons_linked.html")


    import plotly.io as pio
    if add_struct_cicle:
        with open('C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/conn_area_pyr_int/results_bin_x_10_bin_y_20_bin_z_17/cicles_map_'+str(threshold_connection)+'.json', 'r') as f:
            fig = pio.from_json(f.read())

    points = go.Scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                          y=f_pos[pos_neuron_list[10]][0::10, 2],
                          z=f_pos[pos_neuron_list[10]][0::10, 3],
                          mode='markers',
                          marker=dict(size=1,
                                      color='gray',
                                      showscale=True, opacity=0.3)
                          ,

                          )

    for i in range(0,n_intervalli):

        fig = go.Figure(data=points)

        if i==0:
            AUX =np.concatenate((fir_neu_pos_tutti[i][:, 1:4],
                             10 * np.ones([fir_neu_pos_tutti[i].__len__(), 1])), 1)
        else:
            AUX = np.concatenate((np.concatenate((fir_neu_pos_tutti[i][:,1:4],
                                              10 * np.ones([fir_neu_pos_tutti[i].__len__(), 1])), 1),
                              np.concatenate((fir_neu_pos_tutti_diff[i][:,1:4],
                                              100 * np.ones([fir_neu_pos_tutti_diff[i].__len__(), 1])), 1)))
        fig.add_scatter3d(x=AUX[:,0],
                          y=AUX[:,1],
                          z=AUX[:,2],
                           mode='markers',
                           marker=dict(color=AUX[:,3],size=1),opacity=0.7)  # ,showscale=True), )
        fig.update_traces(marker_size=1)
        fig.update_layout(title=dict(text="firing neurons in the interval " + str(iniz_intervals[i]) + " " + str(fine_intervals[i])))
        fig.write_html(path + "scatter_time" + str(i) + ".html")

try:
    #with open(path + 'clusters_eps'+str(eps)+'_m_samp'+str(m_sam)+'.pkl', 'rb') as f:
    #    [intersezioni,center_intersezioni,clusters_diff,clusters,cluster_centers_diff,cluster_centers_diff_weighted,cluster_centers,cluster_centers_weighted] = pickle.load(f)

    with open(path + 'solo_clusters_eps'+str(eps)+'_m_samp'+str(m_sam)+'.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
        [clusters_no_more,clusters_diff, cluster_centers_no_more,clusters,cluster_centers_diff,cluster_centers_diff_weighted,cluster_centers,cluster_centers_weighted] = pickle.load(f)

    print("clusters loaded")
except:
    print("computing clusters ....")
    clusters_diff=np.empty((n_intervalli,), dtype=object)
    clusters_no_more = np.empty((n_intervalli,), dtype=object)
    clusters=np.empty((n_intervalli,), dtype=object)
    cluster_centers_no_more = np.empty((n_intervalli,), dtype=object)
    cluster_centers_diff = np.empty((n_intervalli,), dtype=object)
    cluster_centers_diff_weighted = np.empty((n_intervalli,), dtype=object)
    cluster_centers = np.empty((n_intervalli,), dtype=object)
    cluster_centers_weighted = np.empty((n_intervalli,), dtype=object)
    clusters_diff_founded = False
    clusters_founded = False

    points = go.Scatter3d(x=f_pos[pos_neuron_list[10]][0::10, 1],
                          y=f_pos[pos_neuron_list[10]][0::10, 2],
                          z=f_pos[pos_neuron_list[10]][0::10, 3],
                          mode='markers',
                          marker=dict(size=1,
                                      color='gray', opacity=0.3)
                          ,

                          )

    for i in range(0,n_intervalli):


        #salvo in AUX posizioni e numero di spike dei neuroni che si attivano nell'intervallo i ma non in i-1
        AUX_diff =fir_neu_pos_tutti_diff[i]
        # salvo in AUX posizioni e numero di spike dei neuroni che si attivano nell'intervallo i
        AUX = fir_neu_pos_tutti[i]

        AUX_no_more = fir_neu_pos_tutti_no_more[i]

        if save_all_images:

            if i>0:
                fig = go.Figure(data=points)
                fig.add_scatter3d(x=AUX_diff[:,1],
                              y=AUX_diff[:,2],
                              z=AUX_diff[:,3],
                              mode='markers',
                              marker=dict(color="yellow",size=1),opacity=0.7)# ,showscale=True), )
                #fig.update_traces(marker_size=2)
                fig.update_layout(title = dict(text="new firing neurons in the interval "+ str(iniz_intervals[i])+ " "+ str(fine_intervals[i])))

                fig.write_html(path + "scatter_time" + str(i) + "_only_diff.html")

                fig = go.Figure(data=points)
                fig.add_scatter3d(x=AUX_no_more[:, 1],
                                  y=AUX_no_more[:, 2],
                                  z=AUX_no_more[:, 3],
                                  mode='markers',
                                  marker=dict(color="red", size=1), opacity=0.7)  # ,showscale=True), )
                # fig.update_traces(marker_size=2)
                fig.update_layout(title=dict(
                    text="no more firing neurons in the interval " + str(iniz_intervals[i]) + " " + str(fine_intervals[i])))

                fig.write_html(path + "scatter_time" + str(i) + "_no_more.html")
        #calcolo i clusters e i loro centri per ogni intervallo



        center_cluster = []
        center_cluster_weighted = []

        if AUX.__len__()>0:
            fig = go.Figure(data=points)
            db = DBSCAN(eps=eps, min_samples=m_sam).fit(AUX[:, 1:4])
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            AUX = AUX[labels >= 0, :]
            labels=labels[labels >= 0]
            clusters[i]=np.concatenate((AUX,labels.reshape([labels.__len__(),1])),axis=1)


            if AUX.__len__() > 0:

                fig.add_scatter3d(x=AUX[:, 1],
                              y=AUX[:, 2],
                              z=AUX[:, 3],
                              mode='markers',
                              marker=dict(color=labels, size=3), opacity=0.7)  # ,showscale=True), )


                for j in range(n_clusters_):

                    center_cluster.append(AUX[labels==j,1:4].mean(axis=0))
                    center_cluster_weighted.append(np.average(AUX[labels==j,1:4],axis=0,weights=AUX[labels==j,4]))

                fig.add_scatter3d(x=np.array(center_cluster)[:,0],
                                  y=np.array(center_cluster)[:,1],
                                  z=np.array(center_cluster)[:,2],
                                  mode='markers',
                                  marker=dict(color='darkred', size=3,symbol="x"), opacity=0.7)  # ,showscale=True), )

                fig.add_scatter3d(x=np.array(center_cluster_weighted)[:, 0],
                                  y=np.array(center_cluster_weighted)[:, 1],
                                  z=np.array(center_cluster_weighted)[:, 2],
                                  mode='markers',
                                  marker=dict(color='green', size=3, symbol="x"), opacity=0.7)
                fig.write_html(path_cluster + "scatter_time_" + str(i) + "_clustered.html")
                cluster_centers[i] = center_cluster
                cluster_centers_weighted[i] = center_cluster_weighted
        else:
            clusters[i] = np.empty(0)






        center_cluster_diff=[]
        center_cluster_diff_weighted=[]




        if AUX_diff.__len__()>0:
            fig = go.Figure(data=points)
            db = DBSCAN(eps=eps, min_samples=m_sam).fit(AUX_diff[:,1:4])
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            AUX_diff = AUX_diff[labels >= 0, :]
            labels=labels[labels >= 0]
            clusters_diff[i] = np.concatenate((AUX_diff,labels.reshape([labels.__len__(),1])),axis=1)


            if AUX_diff.__len__() > 0:

                fig.add_scatter3d(x=AUX_diff[:, 1],
                              y=AUX_diff[:, 2],
                              z=AUX_diff[:, 3],
                              mode='markers',
                              marker=dict(color=labels, size=3), opacity=0.7)  # ,showscale=True), )


                for j in range(n_clusters_):

                    center_cluster_diff.append(AUX_diff[labels==j,1:4].mean(axis=0))
                    center_cluster_diff_weighted.append(np.average(AUX_diff[labels==j,1:4],axis=0,weights=AUX_diff[labels==j,4]))

                fig.add_scatter3d(x=np.array(center_cluster_diff)[:,0],
                                  y=np.array(center_cluster_diff)[:,1],
                                  z=np.array(center_cluster_diff)[:,2],
                                  mode='markers',
                                  marker=dict(color='darkred', size=3,symbol="x"), opacity=0.7)  # ,showscale=True), )

                fig.add_scatter3d(x=np.array(center_cluster_diff_weighted)[:, 0],
                                  y=np.array(center_cluster_diff_weighted)[:, 1],
                                  z=np.array(center_cluster_diff_weighted)[:, 2],
                                  mode='markers',
                                  marker=dict(color='green', size=3, symbol="x"), opacity=0.7)
                fig.write_html(path_cluster + "scatter_time" + str(i) + "_only_diff_clustered.html")
                cluster_centers_diff[i] = center_cluster_diff
                cluster_centers_diff_weighted[i] = center_cluster_diff_weighted
        else:
            clusters_diff[i]=np.empty(0)

        center_cluster_no_more = []

        if AUX_no_more.__len__() > 0:
            fig = go.Figure(data=points)
            db = DBSCAN(eps=eps, min_samples=m_sam).fit(AUX_no_more[:, 1:4])
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            AUX_no_more = AUX_no_more[labels >= 0, :]
            labels = labels[labels >= 0]
            clusters_no_more[i] = np.concatenate((AUX_no_more, labels.reshape([labels.__len__(), 1])), axis=1)

            if AUX_no_more.__len__() > 0:

                fig.add_scatter3d(x=AUX_no_more[:, 1],
                                  y=AUX_no_more[:, 2],
                                  z=AUX_no_more[:, 3],
                                  mode='markers',
                                  marker=dict(color=labels, size=3), opacity=0.7)  # ,showscale=True), )

                for j in range(n_clusters_):
                    center_cluster_no_more.append(AUX_no_more[labels == j, 1:4].mean(axis=0))

                fig.add_scatter3d(x=np.array(center_cluster_no_more)[:, 0],
                                  y=np.array(center_cluster_no_more)[:, 1],
                                  z=np.array(center_cluster_no_more)[:, 2],
                                  mode='markers',
                                  marker=dict(color='darkred', size=3, symbol="x"), opacity=0.7)  # ,showscale=True), )


                fig.write_html(path_cluster + "scatter_time" + str(i) + "_no_more_clustered.html")
                cluster_centers_no_more[i] = center_cluster_no_more

        else:
            clusters_no_more[i] = np.empty(0)

    with open(path + 'solo_clusters_eps'+str(eps)+'_m_samp'+str(m_sam)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([clusters_no_more,clusters_diff,cluster_centers_no_more,clusters,cluster_centers_diff, cluster_centers_diff_weighted,cluster_centers,cluster_centers_weighted], f)

try:
    with open(path + 'clusters_eps'+str(eps)+'_m_samp'+str(m_sam)+'.pkl', 'rb') as f:
        [intersezioni,center_intersezioni,intersezioni_out,center_intersezioni_out,clusters_diff,clusters,cluster_centers_diff,cluster_centers_diff_weighted,cluster_centers,cluster_centers_weighted] = pickle.load(f)

    print("intersections loaded")
except:


    intersezioni=np.empty((n_intervalli,), dtype=object)
    center_intersezioni=np.empty((n_intervalli,), dtype=object)
    intersezioni_out = np.empty((n_intervalli,), dtype=object)
    center_intersezioni_out=np.empty((n_intervalli,), dtype=object)

    conn_from_PC=f_pyr[pyr_connection_list[0]][:]
    for i in range (in_connection_list.__len__()):
        if in_connection_list[i][0:5] == "SP_PC":

            conn_from_PC =np.concatenate((conn_from_PC,f_in[in_connection_list[i]][:]))
    try:
        with open('connections_lists.pkl', 'rb') as f:
            [in_conn,out_conn] = pickle.load(f)

    except:

        in_conn = np.empty((int(conn_from_PC[:, 1].max()) + 1,), dtype=object)
        out_conn =np.empty((int(conn_from_PC[:, 0].max()) + 1,), dtype=object)
        for i in range(conn_from_PC[:, 1].__len__()):
            print(i)
            if in_conn[int(conn_from_PC[i, 1])] == None:
                in_conn[int(conn_from_PC[i, 1])] = []
            if out_conn[int(conn_from_PC[i, 0])] == None:
                out_conn[int(conn_from_PC[i, 0])] = []

            in_conn[int(conn_from_PC[i, 1])].append(conn_from_PC[i, 0])
            out_conn[int(conn_from_PC[i, 0])].append(conn_from_PC[i, 1])

        with open('connections_lists.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([in_conn,out_conn], f)


    cl_di_input = Parallel(n_jobs=-1, verbose=50,require='sharedmem')(delayed(find_stimulation_neurons3)(i) for i in range(1,n_intervalli))
    cl_di_input.insert(0,np.empty(0))

    for i in range(1, n_intervalli):
        if cl_di_input[i].__len__() != 0:
            fn=fir_neu_pos_tutti[i-1]
            inters = np.empty((cl_di_input[i].__len__(),), dtype=object)
            centroide = np.empty((cl_di_input[i].__len__(),), dtype=object)
            for j in range(cl_di_input[i].__len__()):
                print(i,j)

                inters[j]=fn[np.isin(fn[:,0],np.unique(cl_di_input[i][j])),:4]

                if inters[j].__len__()!=0:
                    centroide[j]=inters[j][:,1:4].mean(axis=0)
                else:

                    #cl_pre = clusters_diff[i][clusters_diff[i][:, 6] != j, :]
                    #inters[j] = cl_pre[np.isin(cl_pre[:, 0].astype(int), np.unique(cl_di_input[i][j])), :]
                    ind_cl = clusters_diff[i][clusters_diff[i][:, 5] == j, 0]
                    fn =fir_neu_pos_tutti[i]
                    neu_non_in_cl = fn[np.isin(fn[:, 0], ind_cl, invert=True), :]
                    inters[j]=neu_non_in_cl[np.isin(neu_non_in_cl[:, 0].astype(int), np.unique(cl_di_input[i][j])), :]
                    centroide[j] = inters[j][:, 1:4].mean(axis=0)
            intersezioni[i]=inters
            center_intersezioni[i]=centroide

    cl_di_output = Parallel(n_jobs=-1, verbose=50,require='sharedmem')(delayed(find_neurons_stimulated)(i) for i in range(1,n_intervalli))
    cl_di_output.insert(0,np.empty(0))

    for i in range(1, n_intervalli):
         if cl_di_output[i].__len__() != 0:
             fn=fir_neu_pos_tutti[i]
             inters = np.empty((cl_di_output[i].__len__(),), dtype=object)
             centroide = np.empty((cl_di_output[i].__len__(),), dtype=object)
             for j in range(cl_di_output[i].__len__()):
                 print(i,j)

                 inters[j]=fn[np.isin(fn[:,0],np.unique(cl_di_output[i][j])),:4]
                 if inters[j].__len__()!=0:
                     centroide[j]=inters[j][:,1:4].mean(axis=0)
                 else:

                     centroide[j] = posizioni[10][cl_di_output[4][0][cl_di_output[4][0]<n_pyr].astype(int)+1,1:4].mean(axis=0)
             intersezioni_out[i]=inters
             center_intersezioni_out[i]=centroide


    for i in range(1, n_intervalli):
        if cl_di_input[i].__len__() != 0:
            fig = go.Figure(data=points)
            for j in range(intersezioni[i].__len__()):
                fig.add_scatter3d(x=intersezioni[i][j][:, 1],
                                  y=intersezioni[i][j][:, 2],
                                  z=intersezioni[i][j][:, 3],
                                  mode='markers',
                                  marker=dict(color=j, size=3), opacity=0.7)  # ,showscale=True), )

            fig.write_html(path_cluster + "intersezione_"+str(i)+".html")

    for i in range(1, n_intervalli):
        if cl_di_output[i].__len__() != 0:
            fig = go.Figure(data=points)
            for j in range(intersezioni_out[i].__len__()):
                fig.add_scatter3d(x=intersezioni_out[i][j][:, 1],
                                  y=intersezioni_out[i][j][:, 2],
                                  z=intersezioni_out[i][j][:, 3],
                                  mode='markers',
                                  marker=dict(color=j, size=3), opacity=0.7)  # ,showscale=True), )

            fig.write_html(path_cluster + "intersezione_out_"+str(i)+".html")

    with open(path + 'clusters_eps'+str(eps)+'_m_samp'+str(m_sam)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([intersezioni,center_intersezioni,intersezioni_out,center_intersezioni_out,clusters_diff,clusters,cluster_centers_diff,cluster_centers_diff_weighted,cluster_centers,cluster_centers_weighted], f)




n_int_plot=1

centx = []
centy = []
centz = []

x_coni = []
y_coni = []
z_coni = []
u_coni = []
v_coni = []
w_coni = []
line_x = []
line_y = []
line_z = []
time_event = []

centx_between = []
centy_between = []
centz_between = []

x_coni_between = []
y_coni_between = []
z_coni_between = []
u_coni_between = []
v_coni_between = []
w_coni_between = []
line_x_between = []
line_y_between = []
line_z_between = []
time_event_between = []


centx_nm = []
centy_nm = []
centz_nm = []

x_coni_nm = []
y_coni_nm = []
z_coni_nm = []
u_coni_nm = []
v_coni_nm = []
w_coni_nm = []
line_x_nm = []
line_y_nm = []
line_z_nm = []
time_event_nm = []

centx_between_nm = []
centy_between_nm = []
centz_between_nm = []

x_coni_between_nm = []
y_coni_between_nm = []
z_coni_between_nm = []
u_coni_between_nm = []
v_coni_between_nm = []
w_coni_between_nm = []
line_x_between_nm = []
line_y_between_nm = []
line_z_between_nm = []
time_event_between_nm = []



fig_all = go.Figure(data=points)
fig_all_between = go.Figure(data=points)

fig = go.Figure(data=points)
fig_between = go.Figure(data=points)

fig_nm = go.Figure(data=points)
fig_nm_between = go.Figure(data=points)


for i in range(1, n_intervalli):
    fig_all_between = go.Figure(data=points)
    if clusters_diff[i].__len__() != 0:
        lung=intersezioni[i].__len__()
        for j in range(lung):
            x_coni.append(cluster_centers_diff[i][j][0])
            y_coni.append(cluster_centers_diff[i][j][1])
            z_coni.append(cluster_centers_diff[i][j][2])
            u, v, z = (cluster_centers_diff[i][j] - center_intersezioni[i][j]) / np.linalg.norm(
                cluster_centers_diff[i][j] - center_intersezioni[i][j])
            u_coni.append(u)
            v_coni.append(v)
            w_coni.append(z)

            fig_all.add_cone(x=[cluster_centers_diff[i][j][0]],
                         y=[cluster_centers_diff[i][j][1]],
                         z=[cluster_centers_diff[i][j][2]],
                         u=[cluster_centers_diff[i][j][0] - center_intersezioni[i][j][0]],
                         v=[cluster_centers_diff[i][j][1] - center_intersezioni[i][j][1]],
                         w=[cluster_centers_diff[i][j][2] - center_intersezioni[i][j][2]],
                         sizemode="absolute",
                         sizeref=30,

                         anchor="tip"
                         )

            fig.add_cone(x=[cluster_centers_diff[i][j][0]],
                         y=[cluster_centers_diff[i][j][1]],
                         z=[cluster_centers_diff[i][j][2]],
                         u=[cluster_centers_diff[i][j][0] - center_intersezioni[i][j][0]],
                         v=[cluster_centers_diff[i][j][1] - center_intersezioni[i][j][1]],
                         w=[cluster_centers_diff[i][j][2] - center_intersezioni[i][j][2]],
                         sizemode="absolute",
                         sizeref=30,

                         anchor="tip"
                         )

            fig_all_between.add_cone(x=[cluster_centers_diff[i][j][0]],
                                 y=[cluster_centers_diff[i][j][1]],
                                 z=[cluster_centers_diff[i][j][2]],
                                 u=[cluster_centers_diff[i][j][0] - center_intersezioni[i][j][0]],
                                 v=[cluster_centers_diff[i][j][1] - center_intersezioni[i][j][1]],
                                 w=[cluster_centers_diff[i][j][2] - center_intersezioni[i][j][2]],
                                 sizemode="absolute",
                                 sizeref=30,

                                 anchor="tip"
                                 )

            fig_between.add_cone(x=[cluster_centers_diff[i][j][0]],
                         y=[cluster_centers_diff[i][j][1]],
                         z=[cluster_centers_diff[i][j][2]],
                         u=[cluster_centers_diff[i][j][0] - center_intersezioni[i][j][0]],
                         v=[cluster_centers_diff[i][j][1] - center_intersezioni[i][j][1]],
                         w=[cluster_centers_diff[i][j][2] - center_intersezioni[i][j][2]],
                         sizemode="absolute",
                         sizeref=30,

                         anchor="tip"
                         )

            line_x.append(center_intersezioni[i][j][0])
            line_x.append(cluster_centers_diff[i][j][0])
            line_x.append(None)

            line_y.append(center_intersezioni[i][j][1])
            line_y.append(cluster_centers_diff[i][j][1])
            line_y.append(None)

            line_z.append(center_intersezioni[i][j][2])
            line_z.append(cluster_centers_diff[i][j][2])
            line_z.append(None)


            centx.append(center_intersezioni[i][j][0])
            centx.append(cluster_centers_diff[i][j][0])
            centy.append(center_intersezioni[i][j][1])
            centy.append(cluster_centers_diff[i][j][1])
            centz.append(center_intersezioni[i][j][2])
            centz.append(cluster_centers_diff[i][j][2])
            time_event.append(i)
            time_event.append(i)

            line_x_between.append(center_intersezioni[i][j][0])
            line_x_between.append(cluster_centers_diff[i][j][0])
            line_x_between.append(None)

            line_y_between.append(center_intersezioni[i][j][1])
            line_y_between.append(cluster_centers_diff[i][j][1])
            line_y_between.append(None)

            line_z_between.append(center_intersezioni[i][j][2])
            line_z_between.append(cluster_centers_diff[i][j][2])
            line_z_between.append(None)

            centx_between.append(center_intersezioni[i][j][0])
            centx_between.append(cluster_centers_diff[i][j][0])
            centy_between.append(center_intersezioni[i][j][1])
            centy_between.append(cluster_centers_diff[i][j][1])
            centz_between.append(center_intersezioni[i][j][2])
            centz_between.append(cluster_centers_diff[i][j][2])

            time_event_between.append(i)
            time_event_between.append(i)

    if i%n_int_plot==0:

        fig3=fig
        fig3.add_scatter3d(x=line_x,
                          y=line_y,
                          z=line_z,
                          mode='lines',
                          marker=dict(color="green", size=1), opacity=0.7)

        fig3.add_scatter3d(x=centx,
                          y=centy,
                          z=centz,
                          mode='markers',
                          marker=dict(color=time_event, size=3, colorbar=dict(thickness=20)), opacity=0.7)

        fig3.write_html(path_sig + 'signal_trasmission' + str(m_sam) + '_' + str(eps) +'_time_'+str(time[i]) + '.html')


        fig_between.add_scatter3d(x=line_x_between,
                           y=line_y_between,
                           z=line_z_between,
                           mode='lines',
                           marker=dict(color="green", size=1), opacity=0.7)

        fig_between.add_scatter3d(x=centx_between,
                           y=centy_between,
                           z=centz_between,
                           mode='markers',
                           marker=dict(color=time_event_between, size=3, colorbar=dict(thickness=20)), opacity=0.7)

        fig_between.write_html(
            path_sig + 'signal_trasmission' + str(m_sam) + '_' + str(eps) + '_between_' + str(time[i-n_int_plot])+'_and_' + str(time[i]) + '.html')

        fig_all_between.add_scatter3d(x=line_x_between,
                                  y=line_y_between,
                                  z=line_z_between,
                                  mode='lines',
                                  marker=dict(color="green", size=1), opacity=0.7)

        fig_all_between.add_scatter3d(x=centx_between,
                                  y=centy_between,
                                  z=centz_between,
                                  mode='markers',
                                  marker=dict(color=time_event_between, size=3, colorbar=dict(thickness=20)),
                                  opacity=0.7)
        centx_between = []
        centy_between = []
        centz_between = []

        x_coni_between = []
        y_coni_between = []
        z_coni_between = []
        u_coni_between = []
        v_coni_between = []
        w_coni_between = []
        line_x_between = []
        line_y_between = []
        line_z_between = []
        time_event_between = []
        fig_between = go.Figure(data=points)

for i in range(1, n_intervalli):
    if clusters_no_more[i].__len__() != 0:
        lung=intersezioni_out[i].__len__()
        for j in range(lung):
            x_coni_nm.append(center_intersezioni_out[i][j][0])
            y_coni_nm.append(center_intersezioni_out[i][j][1])
            z_coni_nm.append(center_intersezioni_out[i][j][2])
            u, v, z = (center_intersezioni_out[i][j] - cluster_centers_no_more[i][j] ) / np.linalg.norm(
                center_intersezioni_out[i][j] - cluster_centers_no_more[i][j])
            u_coni_nm.append(u)
            v_coni_nm.append(v)
            w_coni_nm.append(z)

            fig_all.add_cone(x=[center_intersezioni_out[i][j][0]],
                            y=[center_intersezioni_out[i][j][1]],
                            z=[center_intersezioni_out[i][j][2]],
                            u=[center_intersezioni_out[i][j][0] - cluster_centers_no_more[i][j][0]],
                            v=[center_intersezioni_out[i][j][1] - cluster_centers_no_more[i][j][1]],
                            w=[center_intersezioni_out[i][j][2] - cluster_centers_no_more[i][j][2]],
                            sizemode="absolute",
                            sizeref=30,

                            anchor="tip"
                            )

            fig_nm.add_cone(x=[center_intersezioni_out[i][j][0]],
                         y=[center_intersezioni_out[i][j][1]],
                         z=[center_intersezioni_out[i][j][2]],
                         u=[center_intersezioni_out[i][j][0]- cluster_centers_no_more[i][j][0] ],
                         v=[center_intersezioni_out[i][j][1]- cluster_centers_no_more[i][j][1]],
                         w=[center_intersezioni_out[i][j][2]- cluster_centers_no_more[i][j][2]],
                         sizemode="absolute",
                         sizeref=30,

                         anchor="tip"
                         )

            fig_all_between.add_cone(x=[center_intersezioni_out[i][j][0]],
                                    y=[center_intersezioni_out[i][j][1]],
                                    z=[center_intersezioni_out[i][j][2]],
                                    u=[center_intersezioni_out[i][j][0] - cluster_centers_no_more[i][j][0]],
                                    v=[center_intersezioni_out[i][j][1] - cluster_centers_no_more[i][j][1]],
                                    w=[center_intersezioni_out[i][j][2] - cluster_centers_no_more[i][j][2]],
                                    sizemode="absolute",
                                    sizeref=30,

                                    anchor="tip"
                                    )

            fig_nm_between.add_cone(x=[center_intersezioni_out[i][j][0]],
                         y=[center_intersezioni_out[i][j][1]],
                         z=[center_intersezioni_out[i][j][2]],
                         u=[center_intersezioni_out[i][j][0]- cluster_centers_no_more[i][j][0] ],
                         v=[center_intersezioni_out[i][j][1]- cluster_centers_no_more[i][j][1]],
                         w=[center_intersezioni_out[i][j][2]- cluster_centers_no_more[i][j][2]],
                         sizemode="absolute",
                         sizeref=30,

                         anchor="tip"
                         )

            line_x_nm.append(center_intersezioni_out[i][j][0])
            line_x_nm.append(cluster_centers_no_more[i][j][0])
            line_x_nm.append(None)

            line_y_nm.append(center_intersezioni_out[i][j][1])
            line_y_nm.append(cluster_centers_no_more[i][j][1])
            line_y_nm.append(None)

            line_z_nm.append(center_intersezioni_out[i][j][2])
            line_z_nm.append(cluster_centers_no_more[i][j][2])
            line_z_nm.append(None)


            centx_nm.append(center_intersezioni_out[i][j][0])
            centx_nm.append(cluster_centers_no_more[i][j][0])
            centy_nm.append(center_intersezioni_out[i][j][1])
            centy_nm.append(cluster_centers_no_more[i][j][1])
            centz_nm.append(center_intersezioni_out[i][j][2])
            centz_nm.append(cluster_centers_no_more[i][j][2])
            time_event_nm.append(i)
            time_event_nm.append(i)

            line_x_between_nm.append(center_intersezioni_out[i][j][0])
            line_x_between_nm.append(cluster_centers_no_more[i][j][0])
            line_x_between_nm.append(None)

            line_y_between_nm.append(center_intersezioni_out[i][j][1])
            line_y_between_nm.append(cluster_centers_no_more[i][j][1])
            line_y_between_nm.append(None)

            line_z_between_nm.append(center_intersezioni_out[i][j][2])
            line_z_between_nm.append(cluster_centers_no_more[i][j][2])
            line_z_between_nm.append(None)

            centx_between_nm.append(center_intersezioni_out[i][j][0])
            centx_between_nm.append(cluster_centers_no_more[i][j][0])
            centy_between_nm.append(center_intersezioni_out[i][j][1])
            centy_between_nm.append(cluster_centers_no_more[i][j][1])
            centz_between_nm.append(center_intersezioni_out[i][j][2])
            centz_between_nm.append(cluster_centers_no_more[i][j][2])

            time_event_between_nm.append(i)
            time_event_between_nm.append(i)

        if clusters_diff[i].__len__() != 0:
            fig_all=fig


    if i%n_int_plot==0:
        fig4=fig_nm
        fig4.add_scatter3d(x=line_x_nm,
                          y=line_y_nm,
                          z=line_z_nm,
                          mode='lines',
                          marker=dict(color="blue", size=1), opacity=0.7)

        fig4.add_scatter3d(x=centx_nm,
                          y=centy_nm,
                          z=centz_nm,
                          mode='markers',
                          marker=dict(color=time_event_nm, size=3, colorbar=dict(thickness=20)), opacity=0.7)

        fig4.write_html(path_sig + 'signal_trasmission_nm_' + str(m_sam) + '_' + str(eps) +'_time_'+str(time[i]) + '.html')

        fig5=fig_all
        fig5.add_scatter3d(x=line_x_nm,
                           y=line_y_nm,
                           z=line_z_nm,
                           mode='lines',
                           marker=dict(color="blue", size=1), opacity=0.7)

        fig5.add_scatter3d(x=centx_nm,
                           y=centy_nm,
                           z=centz_nm,
                           mode='markers',
                           marker=dict(color=time_event_nm, size=3, colorbar=dict(thickness=20)), opacity=0.7)

        fig5.add_scatter3d(x=line_x,
                           y=line_y,
                           z=line_z,
                           mode='lines',
                           marker=dict(color="green", size=1), opacity=0.7)

        fig5.add_scatter3d(x=centx,
                           y=centy,
                           z=centz,
                           mode='markers',
                           marker=dict(color=time_event, size=3, colorbar=dict(thickness=20)), opacity=0.7)
        fig5.write_html(path_sig + 'ALL_signal_trasmission_' + str(m_sam) + '_' + str(eps) + '_time_' + str(
            time[i]) + '.html')

        fig_nm_between.add_scatter3d(x=line_x_between_nm,
                           y=line_y_between_nm,
                           z=line_z_between_nm,
                           mode='lines',
                           marker=dict(color="blue", size=1), opacity=0.7)

        fig_nm_between.add_scatter3d(x=centx_between_nm,
                           y=centy_between_nm,
                           z=centz_between_nm,
                           mode='markers',
                           marker=dict(color=time_event_between_nm, size=3, colorbar=dict(thickness=20)), opacity=0.7)

        fig_nm_between.write_html(
            path_sig + 'signal_trasmission_nm_' + str(m_sam) + '_' + str(eps) + '_between_' + str(time[i-n_int_plot])+'_and_' + str(i * interval_dim) + '.html')

        fig_all_between.add_scatter3d(x=line_x_between_nm,
                                     y=line_y_between_nm,
                                     z=line_z_between_nm,
                                     mode='lines',
                                     marker=dict(color="blue", size=1), opacity=0.7)

        fig_all_between.add_scatter3d(x=centx_between_nm,
                                     y=centy_between_nm,
                                     z=centz_between_nm,
                                     mode='markers',
                                     marker=dict(color=time_event_between_nm, size=3, colorbar=dict(thickness=20)),
                                     opacity=0.7)

        fig_all_between.write_html(path_sig + 'All_sign_trasm_' + str(m_sam) + '_' + str(eps) + '_between_' + str(time[i - n_int_plot]) + '_and_' + str(time[i]) + '.html')

        centx_between_nm = []
        centy_between_nm = []
        centz_between_nm = []

        x_coni_between_nm = []
        y_coni_between_nm = []
        z_coni_between_nm = []
        u_coni_between_nm = []
        v_coni_between_nm = []
        w_coni_between_nm = []
        line_x_between_nm = []
        line_y_between_nm = []
        line_z_between_nm = []
        time_event_between_nm = []
        fig_nm_between = go.Figure(data=points)
        fig_all_between= go.Figure(data=points)





def plot_stim2(i):
    n_cl = int(clusters_diff[i][:,5].max()) + 1
    out = find_stimulation_neurons2(i)
    fig = go.Figure(data=points)
    for j in range(out.__len__()):
        pos_cl_in = posizioni[10][np.unique(out[j]) - 1, 1:4]
        fig.add_scatter3d(x=pos_cl_in[:, 0],
                          y=pos_cl_in[:, 1],
                          z=pos_cl_in[:, 2],
                          mode='markers',
                          marker=dict(color="yellow", size=1), opacity=1)
    T = fir_neu_pos_tutti[i]
    fig.add_scatter3d(x=T[:, 1],
                      y=T[:, 2],
                      z=T[:, 3],
                      mode='markers',
                      marker=dict(color="blue", size=1), opacity=0.2)
    fig.show()




fig.add_scatter3d(x=line_x,
                  y=line_y,
                  z=line_z,
                  mode='lines',
                  marker=dict(color="green", size=1), opacity=0.7)

fig.add_scatter3d(x=centx,
                  y=centy,
                  z=centz,
                  mode='markers',
                  marker=dict(color=time_event, size=3, colorbar=dict(thickness=20)), opacity=0.7)
fig.write_html(path_sig + 'signal_trasmission'+str(m_sam)+'_'+str(eps)+'.html')


for i in range(n_intervalli):
    fig_firing_neuron= go.Figure(data=points)
    fig_firing_neuron.add_scatter3d(x=fir_neu_pos_tutti[i][:,1],
                                y=fir_neu_pos_tutti[i][:,2],
                                z=fir_neu_pos_tutti[i][:,3],
                                mode='markers',
                                marker=dict(color=fir_neu_pos_tutti[i][:,4], size=1, colorbar=dict(thickness=20)), opacity=0.5)

    fig_firing_neuron.write_html(path_fn + 'firing_neuron_between_time_'+str(iniz_intervals[i])+'_and_'+str(fine_intervals[i])+'.html')

soglia=3
for i in range(n_intervalli):
    fig_firing_neuron= go.Figure(data=points)
    fig_firing_neuron.add_scatter3d(x=fir_neu_pos_tutti[i][fir_neu_pos_tutti[i][:,4]>soglia,1],
                                y=fir_neu_pos_tutti[i][fir_neu_pos_tutti[i][:,4]>soglia,2],
                                z=fir_neu_pos_tutti[i][fir_neu_pos_tutti[i][:,4]>soglia,3],
                                mode='markers',
                                marker=dict(color=fir_neu_pos_tutti[i][fir_neu_pos_tutti[i][:,4]>soglia,4], size=1, colorbar=dict(thickness=20)), opacity=0.5)

    fig_firing_neuron.write_html(path_fn + 'firing_neuron_before_time'+str(fine_intervals[i])+'_soglia_'+str(soglia)+'.html')

    counts, bins = np.histogram(fir_neu_pos_tutti[i][:, 4], n_bins)
    plt.figure()
    plt.stairs(counts, bins)
    plt.title("n spk of firing neuron between time "+str(iniz_intervals[i])+" and "+str(fine_intervals[i]))
    plt.savefig(path_fn + "hist_n_sp_of_firing_neuron_between_time_"+str(iniz_intervals[i])+"_and_"+str(fine_intervals[i])+".png")

