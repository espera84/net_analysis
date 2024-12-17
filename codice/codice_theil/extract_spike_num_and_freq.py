import scipy.io as sio
import numpy as np

def features_extraction(list_of_Matrix_spk,n_pop,simulation_time,bin_size):

    ns_at_time=[]
    ns_pop_at_time=[]
    pr=[]
    pr_IC=[]
    pr_at_time=[]
    pr_at_time_IC=[]
    prod_pr_ns_at_time=[]
    prod_pr_ns_at_time_IC=[]

    campionamento=range(0,simulation_time,bin_size)
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu=0


    mu_IC=np.zeros(n_pop)
    pop_size = []

    for i in range(n_pop):

        ns = []
        ns_pop = []


        pop_size.append(list_of_Matrix_spk[i].shape[0])
        ns_at_time.append(list_of_Matrix_spk[i].cumsum(1)[:,campionamento]) #numbero of spikes of each neuron for each time step
        ns_pop_at_time.append(ns_at_time[i].sum(0)) #numbero of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #numbero of spikes of the whole network for each time step

    for i in range(n_pop):
        print(i)
        t=0
        pr.append(ns_at_time[i] / ns_tot_at_time)
        pr_IC.append(ns_at_time[i]/ns_pop_at_time[i] )
        pr_at_time.append(np.nan_to_num(pr[i],nan=0))   #percentage of spikes for each neuron for each step time (on the whole network)
        pr_at_time_IC.append(np.nan_to_num(pr_IC[i], nan=0))    #percentage of spikes for each neuron for each step time (on neurons population)
        prod_pr_ns_at_time.append(pr_at_time[i] * ns_at_time[i])   #product between spike number  and percentage of spikes for each neuron for each step time (on the whole network)
        prod_pr_ns_at_time_IC.append(pr_at_time_IC[i] * ns_at_time[i])  #product between spike number  and percentage of spikes for each neuron for each step time  (on neurons population)
        mu_IC[i]=prod_pr_ns_at_time_IC[i][:,simulation_time_scaled-1].sum()
        mu=mu+prod_pr_ns_at_time[i][:,simulation_time_scaled-1].sum()

    return [ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu,mu_IC]

def features_extraction_alternative(list_of_Matrix_spk,n_pop,simulation_time,bin_size):

    ns_at_time=[]
    ns_pop_at_time=[]
    pr=[]
    pr_IC=[]
    pr_at_time=[]
    pr_at_time_IC=[]
    prod_pr_ns_at_time=[]
    prod_pr_ns_at_time_IC=[]

    campionamento=range(0,simulation_time,bin_size)
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu_at_time=0

    mu_at_time_IC=[]
    mu_IC=np.zeros(n_pop)
    pop_size = []

    for i in range(n_pop):

        ns = []
        ns_pop = []


        pop_size.append(list_of_Matrix_spk[i].shape[0])
        ns_at_time.append(list_of_Matrix_spk[i].cumsum(1)[:,campionamento]) #numbero of spikes of each neuron for each time step
        ns_pop_at_time.append(ns_at_time[i].sum(0)) #numbero of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #numbero of spikes of the whole network for each time step

        #mu_at_time_IC.append(prod_pr_ns_at_time_IC[i].sum(0))
    for i in range(n_pop):
        print(i)
        t=0
        pr.append(ns_at_time[i] / ns_tot_at_time)
        pr_IC.append(ns_at_time[i]/ns_pop_at_time[i] )
        pr_at_time.append(np.nan_to_num(pr[i],nan=0))   #percentage of spikes for each neuron for each step time (on the whole network)
        pr_at_time_IC.append(np.nan_to_num(pr_IC[i], nan=0))    #percentage of spikes for each neuron for each step time (on neurons population)
        prod_pr_ns_at_time.append(pr_at_time[i] * ns_at_time[i])   #product between spike number  and percentage of spikes for each neuron for each step time (on the whole network)
        prod_pr_ns_at_time_IC.append(pr_at_time_IC[i] * ns_at_time[i])  #product between spike number  and percentage of spikes for each neuron for each step time  (on neurons population)
        mu_at_time_IC.append(prod_pr_ns_at_time_IC[i].sum(0))
        mu_at_time=mu_at_time+prod_pr_ns_at_time[i].sum(0)

    return [ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu_at_time,mu_at_time_IC]




def features_extraction_alternative_for_sing_gini(list_of_Matrix_spk,n_pop,simulation_time,bin_size):

    ns_at_time=[]
    ns_pop_at_time=[]
    pr=[]
    pr_IC=[]
    pr_at_time=[]
    pr_at_time_IC=[]
    prod_pr_ns_at_time=[]
    prod_pr_ns_at_time_IC=[]

    #campionamento=range(0,simulation_time+bin_size,bin_size)
    campionamento = [*range(0, simulation_time - 1, bin_size), simulation_time - 1]
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu_at_time=0
    pop_tot=0

    mu_at_time_IC=[]
    mu_IC=np.zeros(n_pop)
    pop_size = []

    for i in range(n_pop):

        ns = []
        ns_pop = []


        pop_size.append(list_of_Matrix_spk[i].shape[0])
        ns_at_time.append(list_of_Matrix_spk[i].cumsum(1)[:,campionamento]) #number of spikes of each neuron for each time step
        ns_pop_at_time.append(ns_at_time[i].sum(0)) #number of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #number of spikes of the whole network for each time step

        #mu_at_time_IC.append(prod_pr_ns_at_time_IC[i].sum(0))
    for i in range(n_pop):
        print(i)
        t=0
        pr.append(ns_at_time[i] / ns_tot_at_time)
        pr_IC.append(ns_at_time[i]/ns_pop_at_time[i] )
        pr_at_time.append(np.nan_to_num(pr[i],nan=0))   #percentage of spikes for each neuron for each step time (on the whole network)
        pr_at_time_IC.append(np.nan_to_num(pr_IC[i], nan=0))    #percentage of spikes for each neuron for each step time (on neurons population)
        prod_pr_ns_at_time.append(pr_at_time[i] * ns_at_time[i])   #product between spike number  and percentage of spikes for each neuron for each step time (on the whole network)
        prod_pr_ns_at_time_IC.append(pr_at_time_IC[i] * ns_at_time[i])  #product between spike number  and percentage of spikes for each neuron for each step time  (on neurons population)
        mu_at_time_IC.append(ns_at_time[i].mean(0))
        pop_tot=pop_tot+pop_size[i]
    mu_at_time=ns_tot_at_time/pop_tot
    return [ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu_at_time,mu_at_time_IC]


def features_extraction_different_format(spk_list,n_pop,start_time,simulation_time,bin_size,neu_ind_interval):

    ns_at_time=[]
    ns_pop_at_time=[]
    pr=[]
    pr_IC=[]
    pr_at_time=[]
    pr_at_time_IC=[]
    prod_pr_ns_at_time=[]
    prod_pr_ns_at_time_IC=[]

    #campionamento=range(0,simulation_time+bin_size,bin_size)
    campionamento = [*range(start_time, simulation_time - 1, bin_size), simulation_time - 1]
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu_at_time=0
    pop_tot=0

    mu_at_time_IC=[]
    mu_IC=np.zeros(n_pop)
    pop_size = []
    ns_at_time_joined = np.zeros([int(np.max(neu_ind_interval)+1), simulation_time_scaled])
    for j in range(simulation_time_scaled):
        np.add.at(ns_at_time_joined[:, j], spk_list[spk_list[:, 1] < campionamento[j], 0].astype(int), 1)
    ns_at_time=[]
    for i in range(n_pop):

        ns = []
        ns_pop = []


        pop_size.append(neu_ind_interval[i][1]-neu_ind_interval[i][0]+1)
        #np.zeros([pop_size[i], simulation_time_scaled]))
        ns_at_time.append(ns_at_time_joined[neu_ind_interval[i][0]:neu_ind_interval[i][1]+1, :])

        #    ns_at_time.append(list_of_Matrix_spk[i].cumsum(1)[:,campionamento]) #number of spikes of each neuron for each time step
        ns_pop_at_time.append(ns_at_time[i].sum(0)) #number of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #number of spikes of the whole network for each time step

        #mu_at_time_IC.append(prod_pr_ns_at_time_IC[i].sum(0))
    for i in range(n_pop):
        print(i)
        t=0
        pr.append(ns_at_time[i] / ns_tot_at_time)
        pr_IC.append(ns_at_time[i]/ns_pop_at_time[i] )
        pr_at_time.append(np.nan_to_num(pr[i],nan=0))   #percentage of spikes for each neuron for each step time (on the whole network)
        pr_at_time_IC.append(np.nan_to_num(pr_IC[i], nan=0))    #percentage of spikes for each neuron for each step time (on neurons population)
        prod_pr_ns_at_time.append(pr_at_time[i] * ns_at_time[i])   #product between spike number  and percentage of spikes for each neuron for each step time (on the whole network)
        prod_pr_ns_at_time_IC.append(pr_at_time_IC[i] * ns_at_time[i])  #product between spike number  and percentage of spikes for each neuron for each step time  (on neurons population)
        mu_at_time_IC.append(ns_at_time[i].mean(0))
        pop_tot=pop_tot+pop_size[i]
    mu_at_time=ns_tot_at_time/pop_tot
    return [ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu_at_time,mu_at_time_IC]

def features_extraction_different_format_red(spk_list,n_pop,start_time,simulation_time,bin_size,neu_ind_interval):

    ns_pop_at_time=[]

    #campionamento=range(0,simulation_time+bin_size,bin_size)
    campionamento = [*range(start_time, simulation_time - 1, bin_size), simulation_time - 1]
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu_at_time=0
    pop_tot=0

    mu_at_time_IC=[]
    mu_IC=np.zeros(n_pop)
    pop_size = []
    ns_at_time_joined = np.zeros([int(np.max(neu_ind_interval)+1), simulation_time_scaled])
    for j in range(simulation_time_scaled):
        print(j)
        np.add.at(ns_at_time_joined[:, j], spk_list[spk_list[:, 1] < campionamento[j], 0].astype(int), 1)
    ns_at_time=[]
    for i in range(n_pop):
        print(i)
        pop_size.append(neu_ind_interval[i][1]-neu_ind_interval[i][0]+1)
        ns_at_time.append(ns_at_time_joined[neu_ind_interval[i][0]:neu_ind_interval[i][1]+1, :])
        print(i)

        ns_pop_at_time.append(ns_at_time[i].sum(0)) #number of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #number of spikes of the whole network for each time step

    for i in range(n_pop):
        print(i)
        mu_at_time_IC.append(ns_at_time[i].mean(0))
        pop_tot=pop_tot+pop_size[i]
    mu_at_time=ns_tot_at_time/pop_tot
    return [ns_at_time,ns_tot_at_time,ns_pop_at_time,mu_at_time,mu_at_time_IC]

def features_extraction_different_format_red2(spk_list,n_pop,start_time,simulation_time,campionamento,neu_ind_interval):

    ns_pop_at_time=[]

    #campionamento=range(0,simulation_time+bin_size,bin_size)
    #campionamento = [*range(start_time, simulation_time - 1, bin_size), simulation_time - 1]
    simulation_time_scaled=np.shape(campionamento)[0]#int(np.ceil(simulation_time/bin_size))
    ns_tot_at_time=np.zeros([simulation_time_scaled])
    mu_at_time=0
    pop_tot=0

    mu_at_time_IC=[]
    mu_IC=np.zeros(n_pop)
    pop_size = []
    ns_at_time_joined = np.zeros([int(np.max(neu_ind_interval)+1), simulation_time_scaled])
    for j in range(1,simulation_time_scaled):
        print(j)
        np.add.at(ns_at_time_joined[:, j], spk_list[spk_list[:, 1] < campionamento[j], 0].astype(int), 1)
    ns_at_time=[]
    for i in range(n_pop):
        print(i)
        pop_size.append(neu_ind_interval[i][1]-neu_ind_interval[i][0]+1)
        ns_at_time.append(ns_at_time_joined[neu_ind_interval[i][0]:neu_ind_interval[i][1]+1, :])
        print(i)

        ns_pop_at_time.append(ns_at_time[i].sum(0)) #number of spikes of neurons population for each time step
        ns_tot_at_time= ns_tot_at_time+ns_pop_at_time[i] #number of spikes of the whole network for each time step

    for i in range(n_pop):
        print(i)
        mu_at_time_IC.append(ns_at_time[i].mean(0))
        pop_tot=pop_tot+pop_size[i]
    mu_at_time=ns_tot_at_time/pop_tot
    return [ns_at_time,ns_tot_at_time,ns_pop_at_time,mu_at_time,mu_at_time_IC]