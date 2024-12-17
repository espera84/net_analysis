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


def save_raster(id_neuroni,n_cc,n_cl):

    global t_initial_analysis,t_final_analysis,results_sub_path
    #plt.ioff()
    fig33=plt.figure()
    spk=[]
    print("test")
    for id_neuron in np.nditer(id_neuroni):

        #print(id_neuron)
        #plt.scatter(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0],i*np.ones(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0].__len__()))
        spk.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][np.logical_and(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]>t_initial_analysis , spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]<t_final_analysis)])
    print("test2")
    plt.eventplot(spk, linelengths=0.7)
    print("test3")
    #plt.legend(loc="upper right")
    plt.title('raster componente connessa'+str(n_cc)+'_clique_'+str(n_cl))
    plt.xlabel("time (ms)")
    plt.ylabel("neuron")
    plt.savefig(results_sub_path + "r_"+str(n_cc)+"_"+str(n_cl)+".png")
    plt.close(fig33)
    plt.clf()
    print('ao'+str(n_cc)+str(n_cl))
    return spk

def save_raster_all(list_id_neuroni):

    global t_initial_analysis,t_final_analysis,results_sub_path
    spk = []

    color = np.array([])
    plt.figure()
    pos_n=1
    for j in range(list_id_neuroni.__len__()):
        id_neuroni=list_id_neuroni[j]
        first_neuron=True
        #print(j)
        for id_neuron in np.nditer(id_neuroni):
            spk_id=spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]
            spk_id_interval=spk_id[np.logical_and(spk_id>t_initial_analysis , spk_id<t_final_analysis)]
            color=np.concatenate((color,np.ones(spk_id_interval.__len__())*j))
            #plt.scatter(spk_list[pos_spk_id, 1][0],i*np.ones(spk_list[pos_spk_id, 1][0].__len__()))
            #spk.append(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0][np.logical_and(spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]>t_initial_analysis , spk_list[np.where(spk_list[:, 0] == id_neuron), 1][0]<t_final_analysis)])
            spk.append(spk_id_interval)

    for i in range(spk.__len__()):
        if i == 0:
            #cl_spk = np.vstack((np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__()))), np.ones(spk[i].__len__()) * i))
            cl_spk = np.vstack((spk[i], np.ones(spk[i].__len__()) * i))
            #cl_spk = np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))
        else:
            # print(np.ones(spk[i].__len__()))
            #cl_spk = np.concatenate((cl_spk, np.vstack((spk[i], np.arange(pos_n, pos_n + spk[i].__len__())))), axis=1)
            cl_spk = np.concatenate((cl_spk,np.vstack((spk[i], np.ones(spk[i].__len__()) * i))) , axis=1)
        pos_n = pos_n + spk[i].__len__()

    k=0
    #fig = px.scatter(x=cl_spk[0,:], y=cl_spk[1,:],color=color)
    #fig.write_image(results_sub_path + "r_cc.svg")
    #fig.show()

    #
    # plt.eventplot(spk, linelengths=0.7)
    # plt.legend(loc="upper right")
    # plt.title('raster componenti connesse')
    # plt.xlabel("time (ms)")
    # plt.ylabel("neuron")
    # plt.savefig(results_sub_path + "r_cc.svg")
    # plt.clf()
    return cl_spk,color

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


t_initial_analysis=30000
t_final_analysis=40000#0#5000#
interval_dim=5
n_workers = -1
perc_attivi=0.15
soglia_attivi=perc_attivi*(t_final_analysis-t_initial_analysis)/interval_dim
perc_corr=0.3
soglia_di_correlazione=perc_corr*(t_final_analysis-t_initial_analysis)/interval_dim
n_shift=0
n_neuroni=288027
n_neuroni_max_da_selezionare=288027#10000#288027
n_neuroni_min_comp_connessa=30
min_clique_size=20

#work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/ls5-1000_p2p5_i2i1_i2p5_p2i1_a3bb1d41-87b5-4397-8bb3-275a53c5efb6/"
#work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl2-1000_p2p5_i2i1_i2p5_p2i1_263e84f9-50a6-4a90-80fb-75a0017c89ed/"
#work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl9-1000_p2p5_i2i1_i2p5_p2i1_8dc41615-d5ce-489e-9a18-fc8709a8346b/"
#work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl15-1000_p2p5_i2i1_i2p5_p2i1_78b7584c-a27f-4e30-97cd-9345d5417efc/"
#work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
if sys.argv.__len__()>2:
    work_path=sys.argv[1]
    sigla=sys.argv[2]
else:
    work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/sl2-1000_p2p5_i2i1_i2p5_p2i1_263e84f9-50a6-4a90-80fb-75a0017c89ed/"
    work_path = "C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/bkg5hz-30_p2p5_i2i1_i2p5_p2i1_b8731d2a-791a-48ac-8b70-b28d4fa1e0bc/"
    sigla="sl2"
print(work_path)

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
results_path=work_path+"/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"
all_results_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/sim/sim_michele_29_8/all/interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"

try:
    os.mkdir(results_path)
except FileExistsError:
    pass


try:
    os.mkdir(all_results_path)
except FileExistsError:
    pass



try:
    with open(results_path + 'neurons_spiking.pkl', 'rb') as f:
        [is_spking,ns_fn_tutti,fn_tutti] = pickle.load(f)
    print("firing neurons loaded")
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

results_sub_path = results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"
all_results_sub_path = all_results_path + "/soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/"

try:
    os.mkdir(results_sub_path)
except FileExistsError:
    pass


try:
    os.mkdir(all_results_sub_path)
except FileExistsError:
    pass


neuroni_attivi=np.where(is_spking[:].sum(1)>soglia_attivi)[0]
n_neuroni_attivi=neuroni_attivi.__len__()
try:
    with open(results_path + 'data_neurons_distances'+str(soglia_attivi)+'_n_neurons'+str(n_neuroni_attivi)+'.pkl', 'rb') as f:
        [neuroni_attivi,n_neuroni_attivi,dif] = pickle.load(f)
    print("neurons distances loaded")
except:




    if (neuroni_attivi.__len__()>n_neuroni_max_da_selezionare):
        neuroni_attivi=neuroni_attivi[::int(neuroni_attivi.__len__()/n_neuroni_max_da_selezionare)+1]


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

    with open(results_path + 'data_neurons_distances'+str(soglia_attivi)+'_n_neurons'+str(n_neuroni_attivi)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([neuroni_attivi,n_neuroni_attivi,dif] , f)


print("plot histogram distances")

plt.hist(dif[:].reshape(pow(dif.__len__(), 2), 1),30)

try:
    with open(results_sub_path +  'data_cc.pkl', 'rb') as f:
        [mat_di_corr,neurons_correlated,id_comp_connesse,n_comp_connesse,list_of_cliques_cc,labels] = pickle.load(f)
    print("connected components data loaded")
except:
    print("calcola mat di correlazione")

    mat_di_corr=dif<soglia_di_correlazione
    neurons_correlated = np.array(np.where(mat_di_corr == True))
    print("calcola componenti connesse")
    graph = csr_matrix(mat_di_corr)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    prima_comp=True
    list_of_cliques_cc=[]
    n_comp_connesse=0
    G = nx.from_numpy_array(mat_di_corr)
    id_comp_connesse = []
    for i in np.nditer(np.unique(labels)):
        comp_connessa=labels==i
        if comp_connessa.sum()>n_neuroni_min_comp_connessa:

            n_comp_connesse=n_comp_connesse+1
            id_comp_connesse.append(i)

            print(comp_connessa.sum())
            mat_comp_conn = mat_di_corr[comp_connessa, :]
            mat_comp_conn = mat_comp_conn[:, comp_connessa]
            G = nx.from_numpy_array(mat_comp_conn)
            ####out = nx.make_max_clique_graph(G) devi controllare , non sembra restituire il massimo clique
            list_of_cliques=list(nx.find_cliques(G))
            list_of_cliques_cc.append(list_of_cliques)

    with open(results_sub_path +  'data_cc.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mat_di_corr,neurons_correlated,id_comp_connesse,n_comp_connesse,list_of_cliques_cc,labels] , f)

try:
        with open(results_sub_path +  'clique_info.pkl', 'rb') as f:
            [id_neu_cliques, indici_cl, cl_spk, col, Isi_cliques,perc_pyr] = pickle.load(f)
        print("connected components data loaded")
except:

    id_neu_cliques=[]
    for i in range(n_comp_connesse):
        comp_connessa = labels == id_comp_connesse[i]
        l=0
        while(list_of_cliques_cc[i].__len__()>0):
            l=l+1

            n_cl = np.zeros([len(list_of_cliques_cc[i])])
            k = 0
            for j in list_of_cliques_cc[i]:
                n_cl[k] = len(j)
                k = k + 1

            ind_clique_max = np.argmax(n_cl)

            id_neuroni_max_clique=neuroni_attivi[comp_connessa][list_of_cliques_cc[i][ind_clique_max]]
            id_neu_cliques.append(id_neuroni_max_clique)
            print("save "+str(l))
            #save_raster(id_neuroni_max_clique,i,l)

            print("comp separate clique " )
            is_separated_clique = np.empty(list_of_cliques_cc[i].__len__(), dtype=bool)
            for j in range(list_of_cliques_cc[i].__len__()):
                is_separated_clique[j] = list(set(list_of_cliques_cc[i][ind_clique_max]).intersection(list_of_cliques_cc[i][j])).__len__() == 0
            list_of_cliques_cc[i] = list(list_of_cliques_cc[i][m] for m in np.where(is_separated_clique)[0])
            print("separate clique computed")
    try:
        [cl_spk,col]=save_raster_all(id_neu_cliques)
    except:
        cl_spk=[]
        col=[]
    indici_cl=[]
    perc_pyr=[]
    for i in range(id_neu_cliques.__len__()):
        if id_neu_cliques[i].__len__()>min_clique_size:
            ind_max=i
            indici_cl.append(i)
            perc_pyr.append((id_neu_cliques[i] <= 261843).sum() / (id_neu_cliques[i] <= 261843).__len__())




    Isi_cliques=[]
    for l in np.nditer(np.array(indici_cl)):#range(id_neu_cliques.__len__()):
        Isi_cliques.append(compute_ISI(id_neu_cliques[l]))

    with open(results_sub_path +  'clique_info.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([id_neu_cliques,indici_cl,cl_spk,col,Isi_cliques,perc_pyr] , f)

#[id_neu_cliques,indici_cl,cl_spk,col,Isi_cliques]
try:

    posizioni_neuroni=posizioni[0]
    for j in range(1,posizioni.__len__()):
        posizioni_neuroni = np.concatenate((posizioni_neuroni, posizioni[j]))

    l=0
    colori=col[np.in1d(col,indici_cl)]
    for i in np.nditer(np.unique(col[np.in1d(col,indici_cl)])):
        colori[col[np.in1d(col,indici_cl)]==i]=l
        l=l+1

    l = 0
    y_raster = cl_spk[1,np.in1d(col,indici_cl)]
    for i in np.nditer(np.unique(cl_spk[1,np.in1d(col,indici_cl)])):
        y_raster[cl_spk[1,np.in1d(col,indici_cl)] == i] = l
        l = l + 1

    fig = px.scatter(x=cl_spk[0,np.in1d(col,indici_cl)], y=y_raster,color=colori)
    #fig = px.scatter(x=cl_spk[0,np.in1d(col,indici_cl)], y=cl_spk[1,np.in1d(col,indici_cl)],color=col[np.in1d(col,indici_cl)])

        #fig.write_image(results_sub_path + "r_cc.svg")
    fig.show()
    fig.write_html(results_sub_path + "raster.html")
    fig.write_html(all_results_sub_path + "raster_"+sigla+".html")

    points = go.Scatter3d(x=posizioni[10][0::10, 1],
                              y=posizioni[10][0::10, 2],
                              z=posizioni[10][0::10, 3],
                              name='network subsampling ',
                              mode='markers',
                              marker=dict(size=1,
                                          color='gray',
                                          showscale=False, opacity=0.3),
                              )
    layout = go.Layout(margin=dict(l=0,
                                       r=0,
                                       b=0,
                                       t=10)
                           )
    fig3 = go.Figure(data=points, layout=layout)


    for l in np.nditer(np.array(indici_cl)):#range(id_neu_cliques.__len__()):

        neuron_to_plt=np.in1d(posizioni_neuroni[:,0],id_neu_cliques[l])
        fig3.add_scatter3d(x=posizioni_neuroni[neuron_to_plt,1],
                           y=posizioni_neuroni[neuron_to_plt,2],
                           z=posizioni_neuroni[neuron_to_plt,3],
                           mode='markers',
                           name='clique_' + str(l),
                           marker=dict(size=5,
                                       colorscale='pinkyl',
                                       color=float(i / n_comp_connesse),
                                       showscale=False,
                                       )
                           )



    fig3.write_html(results_sub_path + "tn4.html")
    fig3.write_html(all_results_sub_path + "tn"+sigla+".html")


    for i in range(Isi_cliques.__len__()):#np.nditer(np.array(indici_cl)):
        plt.clf()
        fig = plt.hist(Isi_cliques[i][Isi_cliques[i] < 100], 50)
        plt.title('ISI Histogram of ' + str(i) + ' connected component neurons ')
        plt.xlabel("ISI (ms)")
        plt.ylabel("# of ISI")
        plt.savefig(results_sub_path + "hist_ISI_" + str(i) + ".png")

    plt.clf()
    out=[]
    for i in range(Isi_cliques.__len__()):
        out.append(plt.hist(Isi_cliques[i][Isi_cliques[i] < 100], 50)[0])
        plt.clf()
    for i in range(Isi_cliques.__len__()):
        plt.plot(out[i], label="clique "+str(indici_cl[0]))

    plt.legend(loc="upper right")
    plt.title('ISI Histograms ')
    plt.xlabel("ISI (ms)")
    plt.ylabel("# of ISI")
    plt.savefig(results_sub_path +"hist_ISI_All.png")
    plt.savefig(all_results_sub_path +"hist_ISI_"+sigla+".png")
    plt.clf()



    plt.clf()
    out=[]
    for i in range(Isi_cliques.__len__()):
        out.append(plt.hist(Isi_cliques[i][Isi_cliques[i] < 100], 50)[0]/Isi_cliques[i][Isi_cliques[i] < 100].__len__())
        plt.clf()
    for i in range(Isi_cliques.__len__()):
        plt.plot(out[i], label="clique "+str(indici_cl[i]))

    plt.legend(loc="upper right")
    plt.title('ISI Distributions ')
    plt.xlabel("ISI (ms)")
    plt.ylabel("# of ISI")
    plt.savefig(results_sub_path +"hist_ISI_All_norm.png")
    plt.savefig(all_results_sub_path +"hist_ISI_"+sigla+"_norm.png")
    plt.clf()


    plt.clf()
    out=[]
    for i in range(Isi_cliques.__len__()):
        out.append(plt.hist(Isi_cliques[i], 50)[0]/Isi_cliques[i].__len__())
        plt.clf()
    for i in range(Isi_cliques.__len__()):
        plt.plot(out[i], label="clique "+str(indici_cl[i]))

    plt.legend(loc="upper right")
    plt.title('ISI Distributions ')
    plt.xlabel("ISI (ms)")
    plt.ylabel("# of ISI")
    plt.savefig(results_sub_path +"hist_ISI_All_norm2.png")
    plt.clf()

    plt.bar( list(map(str, indici_cl)),perc_pyr,width = 0.4)
    plt.title('percentage of pyramidal neurons ')
    plt.xlabel("clique")
    plt.ylabel("% pyr")
    plt.savefig(results_sub_path +"perc_pyr.png")
    plt.savefig(all_results_sub_path +"perc_pyr_"+sigla+"_norm.png")
    plt.clf()


    df = pd.DataFrame (np.transpose(np.vstack((cl_spk,col))))
    df.to_excel(results_sub_path+"raster.xlsx")
except:
    pass