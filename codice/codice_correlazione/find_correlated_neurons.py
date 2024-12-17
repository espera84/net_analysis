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


def calcola_fn(t):

    global spk_list,fn_tutti,ns_fn_tutti

    fn = spk_list[np.logical_and(iniz_intervals[t] < spk_list[:, 1], spk_list[:, 1] < fine_intervals[t]), 0]

    [aux, n_spk_aux] = np.unique(fn, return_counts=True)

    fn_tutti[t]=aux
    ns_fn_tutti[t]=n_spk_aux

def calcola_dif(i):
    global dif, is_spking, neuroni_attivi,shift
    dif[i, :] = (is_spking[neuroni_attivi[i]][:] ^ is_spking[neuroni_attivi[:]]).sum(1).astype(int)
    for j in range(1,n_shift):
        dif1 = (is_spking[neuroni_attivi[i]][:] ^ np.roll(is_spking[neuroni_attivi[:]], j, axis=1)).sum(1).astype(int)
        shift[i, :] = np.maximum(shift[i, :], j * (dif1 < dif[i, :]))
        dif[i, :] = np.minimum(dif[i, :], dif1)

def compute_ISI(id_neuroni):

    isi=np.array([])
    for id_neuron in np.nditer(id_neuroni):
        print(id_neuron)
        isi=np.concatenate((isi,spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][1:] - spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][:-1]))
        #isi.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][1:] - spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][:-1])

    return isi

def save_raster(id_neuroni,n_cc):

    global t_initial_analysis,t_final_analysis
    plt.figure()
    spk=[]
    for id_neuron in np.nditer(id_neuroni):

        print(id_neuron)
        #plt.scatter(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0],i*np.ones(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0].__len__()))
        spk.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][np.logical_and(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]>t_initial_analysis , spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]<t_final_analysis)])
    plt.eventplot(spk, linelengths=0.7)
    plt.legend(loc="upper right")
    plt.title('raster componente connessa'+str(n_cc))
    plt.xlabel("time (ms)")
    plt.ylabel("neuron")
    plt.savefig(results_sub_path + "raster_cc_"+str(n_cc)+".png")
    plt.clf()
    return spk

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


n_neuron_families=f_pos.keys().__len__()
n_bins=500

t_initial_analysis=25000
t_final_analysis=35000#0#5000#
interval_dim=5
n_workers = -1
soglia_attivi=0.25*(t_final_analysis-t_initial_analysis)/interval_dim
soglia_di_correlazione=0.1*(t_final_analysis-t_initial_analysis)/interval_dim
n_shift=1
n_neuroni=288027
n_neuroni_max_da_selezionare=20000#288027
n_neuroni_min_comp_connessa=30

t_initial_analysis=15000
t_final_analysis=25000#0#5000#
interval_dim=10
n_workers = -1
soglia_attivi=0.25*(t_final_analysis-t_initial_analysis)/interval_dim
soglia_di_correlazione=0.1*(t_final_analysis-t_initial_analysis)/interval_dim
n_shift=0
n_neuroni=288027
n_neuroni_max_da_selezionare=288027#10000#288027
n_neuroni_min_comp_connessa=3



work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_dal_24_5/spot1600_2000_exc10_inh12_1caa4615-d33a-407d-8ed9-9a13a205b318/"
work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg10hz-30_p2p5_i2i1_i2p5_p2i1_92ad9aa1-0088-4769-a8ac-23e046c01a07/"
work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p4_i2i0_i2p05_p2i05_8d2079cc-63be-4294-b772-27f212f28c69/"
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
results_path=work_path+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"_bin_"+str(interval_dim)+"/"


try:
    os.mkdir(results_path)
except FileExistsError:
    pass


try:
    with open(results_path + 'neurons_spiking.pkl', 'rb') as f:
        [is_spking,ns_fn_tutti,fn_tutti] = pickle.load(f)
except:


    #n_pyr=posizioni[10].__len__()


    fn_tutti=np.empty((time.shape[0]), dtype=object)
    ns_fn_tutti=np.empty((time.shape[0]), dtype=object)
    is_spking=np.empty((n_neuroni,time.shape[0]), dtype=bool)


    calcola_fn(0)

    Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_fn)(t) for t in range(1, time.shape[0]))

    for i in range(time.shape[0]):
        if(fn_tutti[i].shape[0]>0):
            is_spking[fn_tutti[i].astype(int), i] = True
            #is_spking[fn_tutti[i][fn_tutti[i]].astype(int),i]=True


    #Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_dif)(i) for i in range(n_pyr))
    with open(results_path + 'neurons_spiking.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([is_spking,ns_fn_tutti,fn_tutti], f)

results_sub_path = results_path + "/soglia_attivi_" + str(soglia_attivi)+"_corr_"+str(soglia_di_correlazione)+"n shift"+str(n_shift)+"/"


try:
    os.mkdir(results_sub_path)
except FileExistsError:
    pass


try:
    with open(results_path +  'data_neurons_distances'+str(soglia_attivi)+'_n_neurons'+str(n_neuroni_attivi)+'.pkl', 'rb') as f:
        [neuroni_attivi,n_neuroni_attivi,dif] = pickle.load(f)
except:


    neuroni_attivi=np.where(is_spking[:].sum(1)>soglia_attivi)[0]

    if (neuroni_attivi.__len__()>n_neuroni_max_da_selezionare):
        neuroni_attivi=neuroni_attivi[::int(neuroni_attivi.__len__()/n_neuroni_max_da_selezionare)+1]

    n_neuroni_attivi=neuroni_attivi.__len__()
    print('num neuroni attivi = ', n_neuroni_attivi )

    dif=np.empty((n_neuroni_attivi,n_neuroni_attivi), dtype=int)
    shift=np.zeros((n_neuroni_attivi,n_neuroni_attivi), dtype=int)
    #fai 2 bin
    # for i in range(n_neuroni_attivi):
    #     print(i)
    #     #nella riga i di dif trovo le distanze dell'i-simo neurone attivo
    #     dif[i, :] = (is_spking[neuroni_attivi[i]][:] ^ is_spking[neuroni_attivi[:]]).sum(1).astype(int)
    #     for j in range(n_shift):
    #         dif1=(is_spking[neuroni_attivi[i]][:] ^ np.roll(is_spking[neuroni_attivi[:]], j, axis=1)).sum(1).astype(int)
    #         shift[i, :] = np.maximum(shift[i, :], j * (dif1 > dif[i, :]))
    #         dif[i, :] =np.minimum(dif[i, :],dif1)
    Parallel(n_jobs=n_workers, verbose=50, require='sharedmem')(delayed(calcola_dif)(i) for i in range(n_neuroni_attivi))

    with open(results_path  + 'data_neurons_distances'+str(soglia_attivi)+'_n_neurons'+str(n_neuroni_attivi)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([neuroni_attivi,n_neuroni_attivi,dif] , f)


print("plot histogram distances")

plt.hist(dif[:].reshape(pow(dif.__len__(), 2), 1),30)

try:
    with open(results_sub_path +  'data_comp_conn.pkl', 'rb') as f:
        [mat_di_corr,neurons_correlated,pos_cc2,Isi_comp_connesse,id_comp_connesse,n_comp_connesse,labels] = pickle.load(f)
except:
    print("calcola mat di correlazione")

    mat_di_corr=dif<soglia_di_correlazione
    neurons_correlated = np.array(np.where(mat_di_corr == True))
    print("calcola componenti connesse")
    graph = csr_matrix(mat_di_corr)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)


    print("calcola pos neuroni componenti connesse")
    prima_comp=True
    n_comp_connesse=0
    id_comp_connesse=[]
    Isi_comp_connesse=[]
    for i in np.nditer(np.unique(labels)):
        comp_connessa=labels==i
        if comp_connessa.sum()>n_neuroni_min_comp_connessa:
            n_comp_connesse=n_comp_connesse+1
            print('calcolo ISI comp connessa',i)

            Isi_comp_connesse.append(compute_ISI(neuroni_attivi[comp_connessa]))
            id_comp_connesse.append(i)
            aux_spk=save_raster(neuroni_attivi[comp_connessa], i)


            for j in range(n_neuron_families):
                if j==0 and prima_comp:
                    prima_comp=False
                    pos_cc = posizioni[j][np.isin(posizioni[j][:, 0], neuroni_attivi[comp_connessa]), 1:]
                    pos_cc2 = np.concatenate((pos_cc, np.ones((len(pos_cc),1))*i),axis=1)
                else:
                    pos_cc=posizioni[j][np.isin(posizioni[j][:, 0], neuroni_attivi[comp_connessa]), 1:]
                    pos_cc2=np.concatenate((pos_cc2,np.concatenate((pos_cc, np.ones((len(pos_cc),1))*i),axis=1) ))

    with open(results_sub_path +  'data_comp_conn.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mat_di_corr,neurons_correlated,pos_cc2,Isi_comp_connesse,id_comp_connesse,n_comp_connesse,labels] , f)

points = go.Scatter3d(x=posizioni[10][0::10, 1],
                      y=posizioni[10][0::10, 2],
                      z=posizioni[10][0::10, 3],
                      name='network subsampling ',
                      mode='markers',
                      marker=dict(size=1,
                                  color='gray',
                                  showscale=False, opacity=0.3),
                      )
layout = go.Layout(margin = dict( l = 0,
                                          r = 0,
                                          b = 0,
                                          t = 10)
                           )
fig2 = go.Figure(data=points,layout=layout)

for i in range(n_comp_connesse):

    fig2.add_scatter3d(x=pos_cc2[pos_cc2[:,3]==id_comp_connesse[i], 0],
                       y=pos_cc2[pos_cc2[:,3]==id_comp_connesse[i], 1],
                       z=pos_cc2[pos_cc2[:,3]==id_comp_connesse[i], 2],
                       mode='markers',
                       name='comp conn '+str(id_comp_connesse[i]),
                       marker=dict(size=2,
                                   colorscale='pinkyl',
                                   color=float(i/n_comp_connesse),#'rgba(255,0,0,0.4)',

                                   showscale=False,
                                   )
                       )

fig2.write_html(results_sub_path + "pos_comp_conn.html")

for i in range(id_comp_connesse.__len__()):
    plt.clf()
    fig = plt.hist(Isi_comp_connesse[i][Isi_comp_connesse[i] < 100], 50)
    plt.title('ISI Histogram of ' + str(i) + ' connected component neurons ')
    plt.xlabel("ISI (ms)")
    plt.ylabel("# of ISI")
    plt.savefig(results_sub_path + "hist_ISI_" + str(i) + ".png")

plt.clf()
out=[]
for i in range(id_comp_connesse.__len__()):
    out.append(plt.hist(Isi_comp_connesse[i][Isi_comp_connesse[i] < 100], 50)[0])
    plt.clf()
for i in range(id_comp_connesse.__len__()):
    plt.plot(out[i], label="connected component "+str(id_comp_connesse[i]))

plt.legend(loc="upper right")
plt.title('ISI Histograms ')
plt.xlabel("ISI (ms)")
plt.ylabel("# of ISI")
plt.savefig(results_sub_path +"hist_ISI_All.png")
plt.clf()
out=[]
for i in range(id_comp_connesse.__len__()):
    out.append(plt.hist(Isi_comp_connesse[i][Isi_comp_connesse[i] < 100], 50)[0]/Isi_comp_connesse[i][Isi_comp_connesse[i] < 100].__len__())
    plt.clf()
for i in range(id_comp_connesse.__len__()):
    plt.plot(out[i], label="connected component "+str(id_comp_connesse[i]))

plt.legend(loc="upper right")
plt.title('ISI Distributions ')
plt.xlabel("ISI (ms)")
plt.ylabel("# of ISI")
plt.savefig(results_sub_path +"hist_ISI_All_norm.png")
plt.clf()



    #fig.show()

traslazioni_cc=[]
for ind_cc in range(n_comp_connesse):
    i=id_comp_connesse[ind_cc]
    comp_connessa=labels==i
    neuroni_attivi[comp_connessa]
    mat_comp_conn=mat_di_corr[comp_connessa,:]
    mat_comp_conn=mat_comp_conn[:,comp_connessa]

    shift_comp_conn=shift[comp_connessa,:]
    shift_comp_conn=shift_comp_conn[:,comp_connessa]

    shift_comp_conn=shift_comp_conn*mat_comp_conn
    #shift_comp_conn.sum(1) #somma sulle righe, la riga i riporta in posizione j di quanto bisogna traslare il "firing" di j in avanti per minimizzare la dist dal "firing" di i

    is_vet_trasl_incompleto=True
    j=0
    while(j<mat_comp_conn.__len__() and is_vet_trasl_incompleto):
        vet_tras_temp_comp_conn=np.empty(shift_comp_conn.__len__(),dtype=object)
        vet_tras_temp_comp_conn_old=np.empty(shift_comp_conn.__len__(),dtype=object)
        vet_tras_temp_comp_conn[mat_comp_conn[j, :]] = shift_comp_conn[j, mat_comp_conn[j, :]]
        vet_tras_temp_comp_conn[mat_comp_conn[ :,j]] = - shift_comp_conn[ mat_comp_conn[ :,j],j]

        while (np.any(vet_tras_temp_comp_conn!=None) and np.any(vet_tras_temp_comp_conn_old != vet_tras_temp_comp_conn)):
            vet_tras_temp_comp_conn_old = vet_tras_temp_comp_conn.copy()
            print(np.where(vet_tras_temp_comp_conn!=None))
            for i in np.nditer(np.where(vet_tras_temp_comp_conn!=None)):
                print(i)
            #vet_tras_temp_comp_conn[mat_comp_conn[mat_comp_conn.sum(1).argmax(), :]] = vet_tras_temp_comp_conn[i]+shift_comp_conn[mat_comp_conn.sum(1).argmax(), mat_comp_conn[mat_comp_conn.sum(1).argmax(), :]]
                vet_tras_temp_comp_conn[np.logical_and(mat_comp_conn[i, :], vet_tras_temp_comp_conn == None)] = vet_tras_temp_comp_conn[i]+shift_comp_conn[i, np.logical_and(mat_comp_conn[i, :], vet_tras_temp_comp_conn == None)]
                vet_tras_temp_comp_conn[np.logical_and(mat_comp_conn[:,i], vet_tras_temp_comp_conn == None)] = vet_tras_temp_comp_conn[i] - shift_comp_conn[ np.logical_and(mat_comp_conn[:,i], vet_tras_temp_comp_conn == None),i]

        is_vet_trasl_incompleto = np.where(vet_tras_temp_comp_conn == None)[0].__len__() > 0
        j=j+1
    traslazioni_cc.append(vet_tras_temp_comp_conn)



for i in range(n_comp_connesse):
    fig2 = go.Figure(data=points, layout=layout)
    for j in range(traslazioni_cc[i].min(),traslazioni_cc[i].max()):
        if (np.any([traslazioni_cc[i] == j])):
            fig2.add_scatter3d(x=pos_cc2[pos_cc2[:, 3] == id_comp_connesse[i], 0][traslazioni_cc[i]==j],#pos_cc2[pos_cc2[:, 3] == id_comp_connesse[i], 0],
                               y=pos_cc2[pos_cc2[:, 3] == id_comp_connesse[i], 1][traslazioni_cc[i]==j],
                               z=pos_cc2[pos_cc2[:, 3] == id_comp_connesse[i], 2][traslazioni_cc[i]==j],
                               mode='markers',
                               name='comp conn '+ str(id_comp_connesse[i])+'bin traslazione'+ str(j-traslazioni_cc[i].min()),
                               marker=dict(size=4,
                                           colorscale='pinkyl',
                                           color=float(j-traslazioni_cc[i].min()/ traslazioni_cc[i].max()-traslazioni_cc[i].min()),  # 'rgba(255,0,0,0.4)',
                                           showscale=False,
                                           )
                               )

    fig2.write_html(results_sub_path + "pos_comp_conn"+str(i)+".html")