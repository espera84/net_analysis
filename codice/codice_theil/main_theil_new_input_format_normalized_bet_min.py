import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red2
from gini_WB import compute_Theil_W_B,compute_Theil
from utility import bin_selection
import matplotlib.pyplot as plt
import pickle
import os
import h5py
import json
from matplotlib.pyplot import cm

setting_file="./sim_configuration.json"
sim_conf = json.load(open('%s'%(setting_file), 'r'))
path = sim_conf['Path']
start_time = sim_conf['Time_setting']['Start_time']
simulation_time= sim_conf['Time_setting']['simulation_time']
bin_size=sim_conf['Time_setting']['temporal_bin_size']
without_zeros=True
int_pyr=True
automatic_bin_selection=False

plt.ioff()

no_NGF=False


filename_pos="positions.hdf5"
f_pos=h5py.File(filename_pos, "r")
neurons_families_list=list(f_pos.keys())
if int_pyr:
    labels = ['Pyr','Int']
    n_pop = 2
else:
    labels=neurons_families_list
    n_pop=neurons_families_list.__len__()

path_results =path+"results/"
if no_NGF:
    path_results = path + "resultno_NGF/"
if without_zeros:
    path_results = path + "results_without_zeros/"

if automatic_bin_selection:
    path_results_data_GT = path_results + "ST" + str(start_time) + '_ST_' + str(simulation_time) + "_BS_automatic/"
else:
    path_results_data_GT=path_results+"ST"+str(start_time)  +'_ST_'+str(simulation_time) +'_BS_'+str(bin_size)+"/"#without_zeros"+str(without_zeros)+"/"

if int_pyr:
    path_results_Theil = path_results_data_GT+"Theil_neu_families " + '-'.join(labels) + "/"
else:
    path_results_Theil = path_results_data_GT+"Theil_neu_families_all/"# + '-'.join(labels) + "/"



pop_size=np.zeros(n_pop)
pop_tot=pop_size.sum()
tot_neu=0
neu_ind_interval=[]
norm_fact=np.zeros(n_pop)


for i in range(n_pop):
    if int_pyr:
        if i==0:
            neu_ind_interval.append([int(f_pos[neurons_families_list[10]][:, 0].min()), int(f_pos[neurons_families_list[10]][:, 0].max())])
        else:
            neu_ind_interval.append([ int(f_pos[neurons_families_list[10]][:, 0].max())+1,int(f_pos[neurons_families_list[0]][:, 0].max())])
    else:
        neu_ind_interval.append([int(f_pos[neurons_families_list[i]][:, 0].min()), int(f_pos[neurons_families_list[i]][:, 0].max())])

    pop_size[i] = neu_ind_interval[i][1] - neu_ind_interval[i][0]
    pop_tot = pop_size.sum()

    norm_fact[i]=np.log(neu_ind_interval[i][1]-neu_ind_interval[i][0]+1)
    tot_neu=tot_neu+neu_ind_interval[i][1]-neu_ind_interval[i][0]+1

list_of_list_spk=[]


try:
    os.mkdir(path_results)
except FileExistsError:
    pass

try:
    os.mkdir(path_results_data_GT)
except FileExistsError:
    pass



try:
    os.mkdir(path_results_Theil)
except FileExistsError:
    pass
try:
    #list_of_list_spk = pd.read_csv(path+"activity.csv", header=None)
    #spk_list = np.array(list_of_list_spk)

    list_of_list_spk = h5py.File(path + "activity_network.hdf5", "r")
    spk_list = list_of_list_spk['spikes'].astype('float64')[:]

    #list_of_Matrix_spk=unify_families(mat_contents,pop,start_time,simulation_time,without_zeros)

    if automatic_bin_selection:
        [intervals,n_intervals_points]=bin_selection(spk_list, start_time, simulation_time)
    else:
        n_intervals_points=int(simulation_time / bin_size )+1
        intervals=np.array(range(n_intervals_points)) * bin_size + start_time



    print("pre features extracted")

    try:
        with open(path_results_Theil+ 'GT_features.pkl', 'rb') as f:
            [ns_at_time, ns_tot_at_time, ns_pop_at_time, mu, mu_IC] = pickle.load(f)
    except:
        #[ns_at_time, ns_tot_at_time, ns_pop_at_time, mu, mu_IC] = features_extraction_different_format_red(spk_list,n_pop,start_time,simulation_time,bin_size,neu_ind_interval)
        [ns_at_time, ns_tot_at_time, ns_pop_at_time, mu, mu_IC] = features_extraction_different_format_red2(spk_list,n_pop,start_time,simulation_time,intervals,neu_ind_interval)

        if no_NGF:
            ns_at_time_no_NGF = np.concatenate((ns_at_time[1][:int(f_pos[neurons_families_list[5]][:, 0].min() - f_pos[neurons_families_list[10]][:, 0].max())],ns_at_time[1][int(f_pos[neurons_families_list[5]][:, 0].max() - f_pos[neurons_families_list[10]][:,0].max()):]))
            ns_at_time[1] = ns_at_time_no_NGF
            pop_tot = 0
            mu_IC = []

            for i in range(ns_at_time.__len__()):
                mu_IC.append(ns_at_time[i].mean(0))
                pop_tot = pop_tot + ns_at_time[i].__len__()
            mu = ns_tot_at_time / pop_tot

        if without_zeros:
            ns_at_time_fn = []
            pop_tot = 0
            mu_IC = []

            for i in range(ns_at_time.__len__()):
                ns_at_time_fn.append(ns_at_time[i][ns_at_time[i][:, -1] > 0, :])
                mu_IC.append(ns_at_time_fn[i].mean(0))
                pop_tot = pop_tot + ns_at_time_fn[i].__len__()
            mu = ns_tot_at_time / pop_tot
            ns_at_time = ns_at_time_fn

        with open(path_results_Theil + 'GT_features.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([ns_at_time, ns_tot_at_time, ns_pop_at_time, mu, mu_IC], f)
    print("post features extracted")

    try:
        with open(path_results_Theil + 'Theil_Analisys_data.pkl', 'rb') as f:
            [Theil_for_pop_at_time, Theil_B_at_time, Theil_B_tot_at_time, Theil_W_tot_at_time, Theil_at_time, Theil_for_pop_at_time_interval, Theil_B_at_time_interval, Theil_B_tot_at_time_interval,Theil_W_tot_at_time_interval, Theil_at_time_interval, s, s_interval] = pickle.load(f)
    except:
        [Theil_for_pop_at_time,Theil_B_at_time,Theil_B_tot_at_time,Theil_W_tot_at_time,Theil_at_time,Theil_for_pop_at_time_interval,Theil_B_at_time_interval,Theil_B_tot_at_time_interval,Theil_W_tot_at_time_interval,Theil_at_time_interval,s,s_interval]=compute_Theil_W_B(ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC)
        with open(path_results_Theil + 'Theil_Analisys_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([Theil_for_pop_at_time,Theil_B_at_time,Theil_B_tot_at_time,Theil_W_tot_at_time,Theil_at_time,Theil_for_pop_at_time_interval,Theil_B_at_time_interval,Theil_B_tot_at_time_interval,Theil_W_tot_at_time_interval,Theil_at_time_interval,s,s_interval], f)


    for i in range(norm_fact.__len__()):
        if i == 0:
            norm_fact_tot = norm_fact[i] * np.nan_to_num(s[i, :],0)
            norm_fact_tot_interval = norm_fact[i] * np.nan_to_num(s_interval[i, :],0)
        else:
            norm_fact_tot = norm_fact_tot + norm_fact[i] * np.nan_to_num(s[i, :],0)
            norm_fact_tot_interval = norm_fact_tot_interval + norm_fact[i] * np.nan_to_num(s_interval[i, :],0)



    Theil_for_pop_at_time_norm=np.zeros(Theil_for_pop_at_time.shape)
    Theil_B_at_time_norm=np.zeros(Theil_B_at_time.shape)
    Theil_B_tot_at_time_norm=np.zeros(Theil_B_tot_at_time.shape)
    Theil_W_tot_at_time_norm=np.zeros(Theil_W_tot_at_time.shape)
    Theil_at_time_norm =np.zeros(Theil_at_time.shape)
    Theil_for_pop_at_time_interval_norm=np.zeros(Theil_for_pop_at_time_interval.shape)
    Theil_B_at_time_interval_norm=np.zeros(Theil_B_at_time_interval.shape)
    Theil_B_tot_at_time_interval_norm=np.zeros(Theil_B_tot_at_time_interval.shape)
    Theil_W_tot_at_time_interval_norm=np.zeros(Theil_W_tot_at_time_interval.shape)
    Theil_at_time_interval_norm =np.zeros(Theil_at_time_interval.shape)



    for i in range(n_intervals_points):

        Theil_for_pop_at_time_norm[:,i]=np.nan_to_num(Theil_for_pop_at_time[:,i]/norm_fact,0)
        Theil_B_at_time_norm[:,i]=np.nan_to_num(Theil_B_at_time[:,i] /norm_fact,0)
        Theil_B_at_time_interval_norm[:,i]=np.nan_to_num(Theil_B_at_time_interval[:,i] /norm_fact,0)

    Theil_at_time_interval_norm =np.nan_to_num(Theil_at_time_interval /norm_fact_tot_interval,0)

    Theil_at_time_norm =np.nan_to_num(Theil_at_time /norm_fact_tot,0)
    Theil_for_pop_at_time_interval_norm=np.nan_to_num(Theil_for_pop_at_time_interval/norm_fact_tot_interval,0)
    Theil_B_tot_at_time_norm=np.nan_to_num(Theil_B_tot_at_time /norm_fact_tot,0)
    Theil_W_tot_at_time_norm=np.nan_to_num(Theil_W_tot_at_time /norm_fact_tot,0)
    Theil_B_tot_at_time_interval_norm = np.nan_to_num(Theil_B_tot_at_time_interval/ norm_fact_tot_interval,0)
    Theil_W_tot_at_time_interval_norm = np.nan_to_num(Theil_W_tot_at_time_interval / norm_fact_tot_interval,0)

    #with open(path_results+'Theil_Analisys_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([labels, start_time,simulation_time,bin_size,spk_list,ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu,mu_IC,Theil_for_pop_at_time_norm,Theil_B_at_time_norm,Theil_B_tot_at_time_norm,Theil_W_tot_at_time_norm,Theil_at_time_norm], f)




    for i in range(n_pop):
        plt.figure();
        plt.plot(intervals[:], Theil_for_pop_at_time_norm[i][:], label = labels[i]  )
        plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
        plt.xlabel('Time (ms)')
        plt.ylabel('Theil index cumulative ')
        plt.savefig(path_results_Theil+'Theil_cumulative_'+labels[i]+".png")




    plt.figure();
    plt.plot(intervals,Theil_B_tot_at_time_norm, label='Theil_B_tot'  )
    plt.plot(intervals,Theil_W_tot_at_time_norm, label='Theil_W_tot'  )
    plt.plot(intervals,Theil_at_time_norm, label='Theil'  )
    plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time (ms)')
    plt.ylabel('Theil index')
    #plt.savefig(path + "Gini_Cumulative_within_between.svg")

    plt.savefig(path_results_Theil + 'Theil_Cumulative_wb.png')


    Theil_for_pop_at_time_for_interval_norm=np.zeros([n_pop,2*(ns_tot_at_time.size)])


    for pop in range(n_pop):
        for time in range(ns_tot_at_time.size):
            Theil_for_pop_at_time_for_interval_norm[pop,2*time]=Theil_for_pop_at_time_interval_norm[pop,time]
            Theil_for_pop_at_time_for_interval_norm[pop, 2 * time+1]=Theil_for_pop_at_time_interval_norm[pop,time]

    for i in range(n_pop):
        plt.figure();
        plt.plot(intervals[np.ceil(np.array(range(2*(len(intervals)-1)))/2).astype(int)],Theil_for_pop_at_time_for_interval_norm[i][2:], label=labels[i]  )
        plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
        plt.xlabel('Time (ms)')
        plt.ylabel('Theil index interval ')
        plt.savefig(path_results_Theil + 'Theil_ind_for_inteval_' + labels[i] + ".png")

    fig, ax1 = plt.subplots()
    #plt.figure().set_figwidth(11);
    color = cm.tab20
    [n_spk_per_interval, bi] = np.histogram(spk_list[:, 1], bins=int(simulation_time / bin_size),
                                            range=[start_time, simulation_time + start_time])
    for i in range(n_pop):

        #ax1=plt.plot(intervals[np.ceil(np.array(range(2*(len(intervals)-1)))/2).astype(int)],Theil_for_pop_at_time_for_interval_norm[i][2:], label=labels[i],color=color.colors[i]  )
        ax1.plot(intervals[np.ceil(np.array(range(2 * (len(intervals) - 1))) / 2).astype(int)],
                 Theil_for_pop_at_time_for_interval_norm[i][2:], label=labels[i], color=color.colors[i])

        #box = ax.get_position()
        #plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.legend(bbox_to_anchor=(0.9, 0.5), loc='upper left', borderaxespad=0.)
        plt.xlabel('Time (ms)')
        plt.ylabel('Theil index interval ')
    plt.savefig(path_results_Theil + "Theil_ind_for_inteval_all.png")


    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Theil index interval ')
    ax1.plot(intervals[np.ceil(np.array(range(2*(len(intervals)-1)))/2).astype(int)],Theil_for_pop_at_time_for_interval_norm[i][2:], label=labels[i],color=color.colors[i] )
    ax1.tick_params(axis='y', labelcolor=color.colors[i])

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    ax2.set_ylabel('n spikes', color=color.colors[3])  # we already handled the x-label with ax1
    ax2.plot(bi[1:], n_spk_per_interval, color=color.colors[3])
    ax2.tick_params(axis='y', labelcolor=color.colors[3])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.savefig(path_results_Theil + "Theil_ind_for_inteval_all_w_sp_trace.png")

    plt.figure();
    plt.plot(intervals,Theil_B_tot_at_time_interval_norm, label='Theil_B_tot'  )
    plt.plot(intervals,Theil_W_tot_at_time_interval_norm, label='Theil_W_tot'  )
    plt.plot(intervals,Theil_at_time_interval_norm, label='Theil'  )
    plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time (ms)')
    plt.ylabel('Theil index interval')
    #plt.savefig(path + "Gini_Cumulative_within_between.svg")

    plt.savefig(path_results_Theil + "Theil_Interval_wb.png")

    plt.figure()
    _ = plt.hist(spk_list[:,1], 1000)
    plt.title("number of spikes for interval")
    plt.savefig(path_results_Theil + "num_spikes_for_interval.png")


    if (not no_NGF and not without_zeros):

        entropy_interval=norm_fact[0]-Theil_for_pop_at_time_interval[0][1:]
        entropy_interval_norm = 1 - Theil_for_pop_at_time_interval_norm[0][1:]
        entropy_cumulative=norm_fact[0]-Theil_for_pop_at_time[0][1:]
        entropy_cumulative_norm = 1- Theil_for_pop_at_time_norm[0][1:]

        plt.figure()
        plt.plot(intervals[1:],  entropy_interval)
        plt.title("entropy for interval")
        plt.savefig(path_results_Theil + "entropy_for_interval.png")
        plt.figure()
        plt.plot(intervals[1:], entropy_interval_norm)
        plt.title("entropy for interval normalized")
        plt.savefig(path_results_Theil + "entropy_for_interval_normalized.png")
        plt.figure()
        plt.plot(intervals[1:], entropy_cumulative)
        plt.title("cumulative entropy")
        plt.savefig(path_results_Theil + "cumulative_entropy.png")
        plt.figure()
        plt.plot(intervals[1:], entropy_cumulative_norm)
        plt.title("cumulative entropy normalized")
        plt.savefig(path_results_Theil + "cumulative_entropy_normalized.png")



except FileExistsError:
    pass