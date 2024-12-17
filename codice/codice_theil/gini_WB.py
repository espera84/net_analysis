import scipy.io as sio
import numpy as np
from joblib import Parallel, delayed,dump,load
import os

def compute_gini_WB(ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu,mu_IC,spezzetta,n_pezzi):
    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size
    dif_abs=[]
    pp_IC=[]
    Gini_W=[]
    Gini_JJ=[]
    Gini_B=np.zeros([simulation_time_scaled,n_pop,n_pop])



    for i in range(n_pop):
        pop_size=len(ns_at_time[i])


        if spezzetta:
            bin_i = np.zeros(n_pezzi+1,int)
            for n in range(n_pezzi):
                bin_i[n]=int(n*int(pop_size / n_pezzi))
            bin_i[n_pezzi]=int(pop_size)

            for time in range(simulation_time_scaled):
                add_gini_w=[]
                add_gini_JJ=[]
                for n in range(n_pezzi):
                    for m in range(n_pezzi):

                        pp_IC_aux=((np.outer(pr_at_time_IC[i][bin_i[n]:bin_i[n+1]-1, time],pr_at_time_IC[i][bin_i[m]:bin_i[m+1]-1, time])))
                        pp_aux=(np.outer(pr_at_time[i][bin_i[n]:bin_i[n+1]-1,time],pr_at_time[i][bin_i[m]:bin_i[m+1]-1,time]))
                        dif_abs_aux=(np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1]-1,time], ns_at_time[i][bin_i[m]:bin_i[m+1]-1, time])))
                        add_gini_w.append((pp_aux * dif_abs_aux).sum() / (2 * mu[time]))
                        add_gini_JJ.append((pp_IC_aux * dif_abs_aux ).sum()/ (2 * mu_IC[i][time]))

                #Gini_W.append(np.sum(add_gini_w,0))
                Gini_JJ.append(np.sum(add_gini_JJ,0))
                Gini_W.append(mu_IC[i][time]*(ns_pop_at_time[i][time]**2)/((ns_tot_at_time[time]**2)*mu[time])*Gini_JJ[-1])

        else:
            for time in range(simulation_time_scaled):
                dif_abs.append((np.abs(np.subtract.outer(ns_at_time[i][:,time],ns_at_time[i][:,time]))))
                pp_IC.append((np.outer(pr_at_time_IC[i][:, time], pr_at_time_IC[i][:, time])))
                Gini_JJ.append((pp_IC[-1]*dif_abs[-1]).sum()/(2*mu_IC[i][time]))
                Gini_W.append(mu_IC[i][time]*(ns_pop_at_time[i][time]**2)/((ns_tot_at_time[time]**2)*mu[time])*Gini_JJ[-1])


    Gini_JH=np.zeros([simulation_time_scaled,n_pop,n_pop])

    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):



                aux_GB=[]
                add_GiniJH=[]
                if spezzetta:
                    pop_size=len(ns_at_time[i])
                    bin_i = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_i[n] = int(n * int(pop_size / n_pezzi))
                    bin_i[n_pezzi] = int(pop_size)

                    pop_size = len(ns_at_time[j])
                    bin_j = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_j[n] = int(n * int(pop_size / n_pezzi))
                    bin_j[n_pezzi] = int(pop_size)
                    for time in range(simulation_time_scaled):
                        add_GiniJH=[]
                        for n in range(n_pezzi):
                            for m in range(n_pezzi):

                                pp_aux = (np.outer(pr_at_time_IC[i][bin_i[n]:bin_i[n+1]-1, time], pr_at_time_IC[j][bin_j[m]:bin_j[m+1]-1,time]))
                                dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1]-1,time], ns_at_time[j][bin_j[m]:bin_j[m+1]-1,time])))
                                add_GiniJH.append((pp_aux * dif_abs_aux).sum())
                        Gini_JH[time][i][j]=np.sum(add_GiniJH)/ (mu_IC[i][time] + mu_IC[j][time])
                else:
                    for time in range(simulation_time_scaled):
                        Gini_JH[time][i][j] = (np.outer(pr_at_time_IC[i][:, time],pr_at_time_IC[j][:, time]) * np.abs(np.subtract.outer(ns_at_time[i][:, time],ns_at_time[j][:, time]))).sum() / (mu_IC[i][time] + mu_IC[j][time])

                for time in range(simulation_time_scaled):
                    Gini_B[time,i,j]=(mu_IC[i][time]+mu_IC[j][time])*(ns_pop_at_time[i][time]*ns_pop_at_time[j][time])/((ns_tot_at_time[time]**2)*mu[time])*Gini_JH[time][i][j]



    Gini_W=np.array(Gini_W).reshape(n_pop,simulation_time_scaled)
    Gini_JJ = np.array(Gini_JJ).reshape(n_pop, simulation_time_scaled)

    Gini=[]
    Gini_B_tot=[]
    for time in range(simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time].sum())
        Gini.append(Gini_B_tot[time] + Gini_W[:, time].sum())


    return [Gini_JH[1:simulation_time_scaled,:,:],Gini_JJ[:,1:simulation_time_scaled],Gini_B[1:simulation_time_scaled,:,:],Gini_W[:,1:simulation_time_scaled],Gini_B_tot[1:simulation_time_scaled],Gini[1:simulation_time_scaled]]


def compute_gini(ns_at_tim,pr_at_tim):

    pr_ns_prod=ns_at_tim[:]*pr_at_tim[:]
    ord = pr_at_tim.argsort();
    ns_at_tim= ns_at_tim[ord]
    pr_ns_prod= pr_ns_prod[ord]
    ns_at_tim =ns_at_tim.cumsum()
    pr_ns_prod=pr_ns_prod.cumsum()
    relpop = ns_at_tim/ns_at_tim[-1]
    relz = pr_ns_prod/pr_ns_prod[-1]
    Gini = 1 - sum((relz[0:len(relz)-1] + relz[1:len(relz)])*np.diff(relpop))

    return Gini

def compute_gini_no_freq(ns_at_tim):
    #ns sarebbe val
    ns_at_tim.sort();

    pop=np.ones(ns_at_tim.__len__())/ns_at_tim.__len__()
    z = ns_at_tim * np.ones(ns_at_tim.__len__()) / ns_at_tim.__len__()
    pop=pop.cumsum()
    relpop =pop/pop[-1]#ns_at_tim/ns_at_tim[-1]

    z = z.cumsum()
    relz=z/z[-1]
    Gini = 1 - sum((relz[0:len(relz)-1] + relz[1:len(relz)])*np.diff(relpop))

    return Gini#[Gini,relpop,relz,pop,z]

def compute_gini_WB_alternative(ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu,mu_IC,spezzetta,n_pezzi):
    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size
    dif_abs=[]
    pp_IC=[]
    Gini_W=np.zeros([n_pop,ns_tot_at_time.size-1])
    Gini_JJ=np.zeros([n_pop,ns_tot_at_time.size-1])
    Gini_B=np.zeros([simulation_time_scaled-1,n_pop,n_pop])



    for pop in range(n_pop):
        pop_size = len(ns_at_time[pop])
        for time in range(simulation_time_scaled - 1):
            Gini_JJ [pop, time] = compute_gini(ns_at_time[pop][:, time + 1], pr_at_time[pop][:, time + 1])
            Gini_W[ pop, time] = (mu_IC[pop][time+1]*(ns_pop_at_time[pop][time+1]**2)/((ns_tot_at_time[time+1]**2)*mu[time+1])*Gini_JJ[pop, time])

    Gini_JH=np.zeros([simulation_time_scaled-1,n_pop,n_pop])

    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):



                aux_GB=[]
                add_GiniJH=[]
                if spezzetta:
                    pop_size=len(ns_at_time[i])
                    bin_i = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_i[n] = int(n * int(pop_size / n_pezzi))
                    bin_i[n_pezzi] = int(pop_size)

                    pop_size = len(ns_at_time[j])
                    bin_j = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_j[n] = int(n * int(pop_size / n_pezzi))
                    bin_j[n_pezzi] = int(pop_size)
                    for time in range(1,simulation_time_scaled):
                        add_GiniJH=[]
                        for n in range(n_pezzi):
                            for m in range(n_pezzi):

                                pp_aux = (np.outer(pr_at_time_IC[i][bin_i[n]:bin_i[n+1]-1, time], pr_at_time_IC[j][bin_j[m]:bin_j[m+1]-1,time]))
                                dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1]-1,time], ns_at_time[j][bin_j[m]:bin_j[m+1]-1,time])))
                                add_GiniJH.append((pp_aux * dif_abs_aux).sum())
                        Gini_JH[time-1][i][j]=np.sum(add_GiniJH)/ (mu_IC[i][time] + mu_IC[j][time])
                else:
                    for time in range(1,simulation_time_scaled):
                        Gini_JH[time-1][i][j] = (np.outer(pr_at_time_IC[i][:, time],pr_at_time_IC[j][:, time]) * np.abs(np.subtract.outer(ns_at_time[i][:, time],ns_at_time[j][:, time]))).sum() / (mu_IC[i][time] + mu_IC[j][time])

                for time in range(1,simulation_time_scaled):
                    Gini_B[time-1,i,j]=(mu_IC[i][time]+mu_IC[j][time])*(ns_pop_at_time[i][time]*ns_pop_at_time[j][time])/((ns_tot_at_time[time]**2)*mu[time])*Gini_JH[time-1][i][j]


    Gini=[]
    Gini_B_tot=[]
    for time in range(1,simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time-1].sum())
        Gini.append(Gini_B_tot[time-1] + Gini_W[:, time-1].sum())


    return [Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini]


def compute_gini_WB_alternative_no_freq(ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC,spezzetta,n_pezzi):
    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size

    Gini_W=np.zeros([n_pop,ns_tot_at_time.size])
    Gini_JJ=np.zeros([n_pop,ns_tot_at_time.size])
    Gini_B=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_B_interval=np.zeros([simulation_time_scaled,n_pop,n_pop])
    pop_size=np.zeros(len(ns_at_time))
    Gini_W_interval=np.zeros([n_pop,ns_tot_at_time.size])
    Gini_JJ_interval=np.zeros([n_pop,ns_tot_at_time.size])

    for i in range(n_pop):
        pop_size[i]=len(ns_at_time[i])

    pop_tot=pop_size.sum()



    for pop in range(n_pop):

        for time in range(simulation_time_scaled):
            #Gini_JJ [pop, time] = compute_gini_no_freq(ns_at_time[pop][:, time + 1])
            #Gini_W[ pop, time] = (mu_IC[pop][time+1]*(ns_pop_at_time[pop][time+1]**2)/((ns_tot_at_time[time+1]**2)*mu[time+1])*Gini_JJ[pop, time])
            #Gini_JJ_interval[pop, time] =  compute_gini_no_freq(ns_at_time[pop][:, time + 1]-ns_at_time[pop][:, time])

            Gini_JJ[pop, time] = compute_gini_no_freq(ns_at_time[pop][:, time])
            Gini_W[ pop, time] = (mu_IC[pop][time]*(ns_pop_at_time[pop][time]**2)/((ns_tot_at_time[time]**2)*mu[time])*Gini_JJ[pop, time])




            if time == 0:
                Gini_JJ_interval[pop, time]=Gini_JJ[pop, time]
                ns_pop_interval = ns_pop_at_time[pop][time]
                mu_IC_interval = ns_pop_interval / pop_size[pop]
                Gini_W_interval[pop, time] = (mu_IC_interval * (ns_pop_interval ** 2) / (
                            ((ns_tot_at_time[time]) ** 2) * (mu[time + 1] - mu[time])) *
                                              Gini_JJ_interval[pop, time])

            else:
                # print("time")
                Gini_JJ_interval[pop, time] = compute_gini_no_freq(ns_at_time[pop][:, time] - ns_at_time[pop][:, time - 1])
                ns_pop_interval = ns_pop_at_time[pop][time] - ns_pop_at_time[pop][time - 1]

                mu_IC_interval = ns_pop_interval / pop_size[pop]
                Gini_W_interval[pop, time] = (mu_IC_interval * (ns_pop_interval ** 2) / (((ns_tot_at_time[time]-ns_tot_at_time[time-1 ]) ** 2) * (mu[time ]-mu[time-1])) * Gini_JJ_interval[pop, time])


    Gini_JH=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_JH_interval = np.zeros([simulation_time_scaled, n_pop, n_pop])
    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):

                if spezzetta:
                    bin_i = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_i[n] = int(n * int(pop_size[i] / n_pezzi))
                    bin_i[n_pezzi] = int(pop_size[i])

                    bin_j = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_j[n] = int(n * int(pop_size[j] / n_pezzi))
                    bin_j[n_pezzi] = int(pop_size[j])
                    for time in range(simulation_time_scaled):
                        add_GiniJH=[]
                        dif_abs_sum = 0
                        dif_abs_sum_interval=0
                        for n in range(n_pezzi):
                            for m in range(n_pezzi):

                                dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1],time], ns_at_time[j][bin_j[m]:bin_j[m+1],time])))
                                add_GiniJH.append(dif_abs_aux.sum())
                                dif_abs_sum = dif_abs_sum + dif_abs_aux.sum()

                                if time == 0:
                                    ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n + 1], time]
                                    ns_interval_j = ns_at_time[i][bin_j[m]:bin_j[m + 1], time]
                                else:
                                    ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n+1],time] - ns_at_time[i][bin_i[n]:bin_i[n+1],time-1]
                                    ns_interval_j = ns_at_time[j][bin_j[m]:bin_j[m + 1] , time] - ns_at_time[j][bin_j[m]:bin_j[m + 1] , time - 1]
                                dif_abs_sum_interval = dif_abs_sum_interval+np.abs(np.subtract.outer(ns_interval_i, ns_interval_j)).sum()

                        if time == 0:
                            ns_pop_interval_i = ns_pop_at_time[i][time]
                            mu_interval = ns_tot_at_time[time] / pop_tot
                            ns_pop_interval_j = ns_pop_at_time[j][time]

                        else:
                            ns_pop_interval_i = ns_pop_at_time[i][time] - ns_pop_at_time[i][time - 1]
                            ns_pop_interval_j = ns_pop_at_time[j][time] - ns_pop_at_time[j][time - 1]
                            mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                        mu_IC_interval_i = ns_pop_interval_i / pop_size[i]
                        mu_IC_interval_j = ns_pop_interval_j / pop_size[j]

                        Gini_JH_interval[time][i][j] = dif_abs_sum_interval / ((mu_IC_interval_i+ mu_IC_interval_j) * pop_size[j] * pop_size[i])
                        Gini_B_interval[time, i, j] = (mu_IC_interval_i + mu_IC_interval_j) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu_interval) * Gini_JH_interval[time][i][j]

                        Gini_JH[time][i][j]=np.sum(add_GiniJH)/ ((mu_IC[i][time] + mu_IC[j][time])*pop_size[j]*pop_size[i])
                else:
                    for time in range(simulation_time_scaled):
                        dif_abs_sum=np.abs(np.subtract.outer(ns_at_time[i][:, time], ns_at_time[j][:, time])).sum()

                        ns_interval_i = ns_at_time[i][:, time] - ns_at_time[i][:, time - 1]
                        ns_interval_j = ns_at_time[j][:, time] - ns_at_time[j][:, time - 1]
                        dif_abs_sum_interval = np.abs(np.subtract.outer(ns_interval_i, ns_interval_j)).sum()


                        if time == 0:
                            ns_pop_interval_i = ns_pop_at_time[i][time]
                            mu_interval = ns_tot_at_time[time] / pop_tot
                            ns_pop_interval_j = ns_pop_at_time[j][time]

                        else:
                            ns_pop_interval_i = ns_pop_at_time[i][time] - ns_pop_at_time[i][time - 1]
                            ns_pop_interval_j = ns_pop_at_time[j][time] - ns_pop_at_time[j][time - 1]
                            mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                        mu_IC_interval_i = ns_pop_interval_i / pop_size[i]
                        mu_IC_interval_j = ns_pop_interval_j / pop_size[j]

                        Gini_JH_interval[time][i][j] = dif_abs_sum_interval / ((mu_IC_interval_i + mu_IC_interval_j) * pop_size[j] * pop_size[i])
                        Gini_B_interval[time, i, j] = (mu_IC_interval_i + mu_IC_interval_j) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu_interval) * Gini_JH_interval[time][i][j]
                        Gini_JH[time][i][j] = dif_abs_sum / ((mu_IC[i][time] + mu_IC[j][time])*pop_size[j]*pop_size[i])

                for time in range(simulation_time_scaled):
                    Gini_B[time,i,j]=(mu_IC[i][time]+mu_IC[j][time])*(pop_size[i]*pop_size[j])/((pop_tot**2)*mu[time])*Gini_JH[time][i][j]



    Gini=[]
    Gini_B_tot=[]
    Gini_B_interval_tot=[]
    Gini_interval=[]
    for time in range(simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time].sum())
        Gini_B_interval_tot.append(Gini_B_interval[time].sum())
        Gini.append(Gini_B_tot[time] + Gini_W[:, time].sum())
        Gini_interval.append(Gini_B_interval_tot[time] + Gini_W_interval[:, time].sum())



    return [Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini,Gini_JJ_interval,Gini_W_interval,Gini_B_interval,Gini_B_interval_tot,Gini_interval]





def compute_gini_WB_without_freq(ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC,spezzetta,n_pezzi):

    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size
    dif_abs=[]
    pp_IC=[]
    Gini_W=[]
    Gini_JJ=[]
    Gini_B=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_B_interval=np.zeros([simulation_time_scaled,n_pop,n_pop])
    pop_size=np.zeros(len(ns_at_time))
    Gini_W_interval=[]
    Gini_JJ_interval=[]


    for i in range(n_pop):
        pop_size[i]=len(ns_at_time[i])

    pop_tot=pop_size.sum()

    for i in range(n_pop):
        print("pop",i)


        if spezzetta:

            bin_i = np.zeros(n_pezzi+1,int)
            for n in range(n_pezzi):
                bin_i[n]=int(n*int(pop_size[i] / n_pezzi))
            bin_i[n_pezzi]=int(pop_size[i])

            dif_abs_sum_prec=0
            for time in range(simulation_time_scaled):
                add_gini_w=[]
                add_gini_JJ=[]

                dif_abs_sum=0
                dif_abs_sum_interval =0
                for n in range(n_pezzi):
                    for m in range(n_pezzi):
                        print("time",time)
                        print("n", n)
                        print("m", m)

                        print("n_pezzi", n_pezzi)

                        dif_abs_aux=(np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1],time], ns_at_time[i][bin_i[m]:bin_i[m+1], time])))
                        add_gini_JJ.append(dif_abs_aux.sum()/ (2 *(pop_size[i]**2) * mu_IC[i][time]))
                        dif_abs_sum = dif_abs_sum+ dif_abs_aux.sum()
                        if time==0:
                            ns_interval_n = ns_at_time[i][bin_i[n]:bin_i[n + 1] , time]
                            ns_interval_m = ns_at_time[i][bin_i[m]:bin_i[m + 1] , time]
                        else:
                            ns_interval_n =ns_at_time[i][bin_i[n]:bin_i[n+1],time]-ns_at_time[i][bin_i[n]:bin_i[n+1],time-1]
                            ns_interval_m =ns_at_time[i][bin_i[m]:bin_i[m + 1] , time] - ns_at_time[i][bin_i[m]:bin_i[m + 1],time - 1]
                        dif_abs_sum_interval = dif_abs_sum_interval + np.abs(np.subtract.outer(ns_interval_n, ns_interval_m)).sum()

                if time==0:
                    ns_pop_interval = ns_pop_at_time[i][time]
                    mu_interval = ns_tot_at_time[time] / pop_tot
                else:
                    ns_pop_interval=ns_pop_at_time[i][time]-ns_pop_at_time[i][time-1]
                    mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                mu_IC_interval=ns_pop_interval/pop_size[i]
                Gini_JJ_interval.append(dif_abs_sum_interval/(2 *(pop_size[i]**2) * mu_IC_interval ))
                Gini_W_interval.append(mu_IC_interval*(pop_size[i]**2)/((pop_tot**2)*mu_interval*Gini_JJ_interval[-1]))
                Gini_JJ.append(np.sum(add_gini_JJ,0))
                Gini_W.append((mu_IC[i][time]*(pop_size[i]**2))/((pop_tot**2)*mu[time])*Gini_JJ[-1])

        else:

            for time in range(simulation_time_scaled):
                print("time", time)
                dif_abs.append(np.abs(np.subtract.outer(ns_at_time[i][:,time],ns_at_time[i][:,time])))
                dif_abs_sum =np.sum(dif_abs[-1])
                ns_interval=ns_at_time[i][:,time]-ns_at_time[i][:,time-1]
                dif_abs_sum_interval =np.abs(np.subtract.outer(ns_interval,ns_interval)).sum()

                if time==0:
                    ns_pop_interval = ns_pop_at_time[i][time]
                    mu_interval = ns_tot_at_time[time] / pop_tot
                else:
                    ns_pop_interval=ns_pop_at_time[i][time]-ns_pop_at_time[i][time-1]
                    mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                mu_IC_interval = ns_pop_interval / pop_size[i]


                Gini_JJ_interval.append(dif_abs_sum_interval / (2 * (pop_size[i] ** 2) * mu_IC_interval))
                Gini_W_interval.append(mu_IC_interval * (pop_size[i] ** 2) / ((pop_tot ** 2) * mu_interval * Gini_JJ_interval[-1]))
                Gini_JJ.append(dif_abs_sum / (2 * (pop_size[i] ** 2) * mu_IC[i][time]))
                Gini_W.append(mu_IC[i][time] * (pop_size[i] ** 2) / ((pop_tot ** 2) * mu[time]) * Gini_JJ[-1])

    Gini_JH=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_JH_interval = np.zeros([simulation_time_scaled, n_pop, n_pop])
    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):



                aux_GB=[]
                add_GiniJH=[]
                if spezzetta:
                    bin_i = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_i[n] = int(n * int(pop_size[i] / n_pezzi))
                    bin_i[n_pezzi] = int(pop_size[i])

                    bin_j = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_j[n] = int(n * int(pop_size[j] / n_pezzi))
                    bin_j[n_pezzi] = int(pop_size[j])
                    for time in range(simulation_time_scaled):
                        add_GiniJH=[]
                        dif_abs_sum = 0
                        dif_abs_sum_interval=0
                        for n in range(n_pezzi):
                            for m in range(n_pezzi):

                                dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1],time], ns_at_time[j][bin_j[m]:bin_j[m+1],time])))
                                add_GiniJH.append(dif_abs_aux.sum())
                                dif_abs_sum = dif_abs_sum + dif_abs_aux.sum()

                                if time == 0:
                                    ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n + 1], time]
                                    ns_interval_j = ns_at_time[i][bin_j[m]:bin_j[m + 1], time]
                                else:
                                    ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n+1],time] - ns_at_time[i][bin_i[n]:bin_i[n+1],time-1]
                                    ns_interval_j = ns_at_time[j][bin_j[m]:bin_j[m + 1] , time] - ns_at_time[j][bin_j[m]:bin_j[m + 1] , time - 1]
                                dif_abs_sum_interval = dif_abs_sum_interval+np.abs(np.subtract.outer(ns_interval_i, ns_interval_j)).sum()

                        if time == 0:
                            ns_pop_interval_i = ns_pop_at_time[i][time]
                            mu_interval = ns_tot_at_time[time] / pop_tot
                            ns_pop_interval_j = ns_pop_at_time[j][time]

                        else:
                            ns_pop_interval_i = ns_pop_at_time[i][time] - ns_pop_at_time[i][time - 1]
                            ns_pop_interval_j = ns_pop_at_time[j][time] - ns_pop_at_time[j][time - 1]
                            mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                        mu_IC_interval_i = ns_pop_interval_i / pop_size[i]
                        mu_IC_interval_j = ns_pop_interval_j / pop_size[j]

                        Gini_JH_interval[time][i][j] = dif_abs_sum_interval / ((mu_IC_interval_i+ mu_IC_interval_j) * pop_size[j] * pop_size[i])
                        Gini_B_interval[time, i, j] = (mu_IC_interval_i + mu_IC_interval_j) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu_interval) * Gini_JH_interval[time][i][j]

                        Gini_JH[time][i][j]=np.sum(add_GiniJH)/ ((mu_IC[i][time] + mu_IC[j][time])*pop_size[j]*pop_size[i])
                else:
                    for time in range(simulation_time_scaled):
                        dif_abs_sum=np.abs(np.subtract.outer(ns_at_time[i][:, time], ns_at_time[j][:, time])).sum()

                        ns_interval_i = ns_at_time[i][:, time] - ns_at_time[i][:, time - 1]
                        ns_interval_j = ns_at_time[j][:, time] - ns_at_time[j][:, time - 1]
                        dif_abs_sum_interval = np.abs(np.subtract.outer(ns_interval_i, ns_interval_j)).sum()


                        if time == 0:
                            ns_pop_interval_i = ns_pop_at_time[i][time]
                            mu_interval = ns_tot_at_time[time] / pop_tot
                            ns_pop_interval_j = ns_pop_at_time[j][time]

                        else:
                            ns_pop_interval_i = ns_pop_at_time[i][time] - ns_pop_at_time[i][time - 1]
                            ns_pop_interval_j = ns_pop_at_time[j][time] - ns_pop_at_time[j][time - 1]
                            mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

                        mu_IC_interval_i = ns_pop_interval_i / pop_size[i]
                        mu_IC_interval_j = ns_pop_interval_j / pop_size[j]

                        Gini_JH_interval[time][i][j] = dif_abs_sum_interval / ((mu_IC_interval_i + mu_IC_interval_j) * pop_size[j] * pop_size[i])
                        Gini_B_interval[time, i, j] = (mu_IC_interval_i + mu_IC_interval_j) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu_interval) * Gini_JH_interval[time][i][j]
                        Gini_JH[time][i][j] = dif_abs_sum / ((mu_IC[i][time] + mu_IC[j][time])*pop_size[j]*pop_size[i])

                for time in range(simulation_time_scaled):
                    Gini_B[time,i,j]=(mu_IC[i][time]+mu_IC[j][time])*(pop_size[i]*pop_size[j])/((pop_tot**2)*mu[time])*Gini_JH[time][i][j]



    Gini_W=np.array(Gini_W).reshape(n_pop,simulation_time_scaled)
    Gini_JJ = np.array(Gini_JJ).reshape(n_pop, simulation_time_scaled)
    Gini_W_interval=np.array(Gini_W_interval).reshape(n_pop,simulation_time_scaled)
    Gini_JJ_interval = np.array(Gini_JJ_interval).reshape(n_pop, simulation_time_scaled)

    Gini=[]
    Gini_B_tot=[]
    Gini_B_interval_tot=[]
    Gini_interval=[]
    for time in range(simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time].sum())
        Gini_B_interval_tot.append(Gini_B_interval[time].sum())
        Gini.append(Gini_B_tot[time] + Gini_W[:, time].sum())
        Gini_interval.append(Gini_B_interval_tot[time] + Gini_W_interval[:, time].sum())




    return [Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini,Gini_JJ_interval,Gini_W_interval,Gini_B_interval,Gini_B_interval_tot,Gini_interval]

def compute_gini_WB_without_freq_parallel(ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC,n_pezzi):

    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size
    dif_abs=[]
    pp_IC=[]
    Gini_W=[]
    Gini_JJ=[]
    Gini_B=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_B_interval=np.zeros([simulation_time_scaled,n_pop,n_pop])
    pop_size=np.zeros(len(ns_at_time))
    Gini_W_interval=[]
    Gini_JJ_interval=[]

    #folder = './joblib_memmap'
    #try:
    #    os.mkdir(folder)
    #except FileExistsError:
    #    pass

    #data_filename_memmap = os.path.join(folder, 'data_memmap')
    #dump(ns_at_time, data_filename_memmap)
    #ns_at_time = load(data_filename_memmap, mmap_mode='r')

    for i in range(n_pop):
        pop_size[i]=len(ns_at_time[i])

    pop_tot=pop_size.sum()

    for pop in range(n_pop):
        print("pop",pop)

        bin_i = np.zeros(n_pezzi+1,int)
        for n in range(n_pezzi):
            bin_i[n]=int(n*int(pop_size[pop] / n_pezzi))
        bin_i[n_pezzi]=int(pop_size[pop])
        num_cores=10
        #aux=[]
        #for time in range(simulation_time_scaled):
        #    print(time)
        #    out= calcola_gini_JJ_W_new2( ns_tot_at_time,pop_tot,pop_size,ns_pop_at_time, mu_IC,mu,pop,time)
        #    aux.append(out)
        aux= Parallel(n_jobs=num_cores, verbose=50)(delayed(calcola_gini_JJ_W_new)(ns_at_time,ns_tot_at_time,pop_tot,pop_size,ns_pop_at_time, mu_IC,mu,pop,time) for time in range(simulation_time_scaled))
        aux1=np.array(aux)
        Gini_JJ_interval.append(aux1[:,0])
        Gini_W_interval.append(aux1[:,1])
        Gini_JJ.append(aux1[:,2])
        Gini_W.append(aux1[:,3])

        dif_abs_sum_prec=0


    Gini_JH=np.zeros([simulation_time_scaled,n_pop,n_pop])
    Gini_JH_interval = np.zeros([simulation_time_scaled, n_pop, n_pop])

    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):



                aux_GB=[]
                add_GiniJH=[]
                bin_i = np.zeros(n_pezzi + 1, int)
                for n in range(n_pezzi):
                    bin_i[n] = int(n * int(pop_size[i] / n_pezzi))
                bin_i[n_pezzi] = int(pop_size[i])

                bin_j = np.zeros(n_pezzi + 1, int)
                for n in range(n_pezzi):
                    bin_j[n] = int(n * int(pop_size[j] / n_pezzi))
                bin_j[n_pezzi] = int(pop_size[j])

                aux2 = Parallel(n_jobs=num_cores, verbose=50,prefer="threads")(delayed(calcola_JH)(ns_at_time, bin_i, bin_j, ns_tot_at_time, pop_tot, n_pezzi, pop_size, ns_pop_at_time, mu_IC,i, j, time,Gini_JH,Gini_B_interval,Gini_JH_interval) for time in range(simulation_time_scaled))
                aux1 = np.array(aux2)

                for time in range(simulation_time_scaled):
                    Gini_B[time, i, j] = (mu_IC[i][time] + mu_IC[j][time]) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu[time]) * Gini_JH[time][i][j]



    Gini_W=np.array(Gini_W)#.reshape(n_pop,simulation_time_scaled)
    Gini_JJ = np.array(Gini_JJ)#.reshape(n_pop, simulation_time_scaled)
    Gini_W_interval=np.array(Gini_W_interval)#.reshape(n_pop,simulation_time_scaled)
    Gini_JJ_interval = np.array(Gini_JJ_interval)#.reshape(n_pop, simulation_time_scaled)

    Gini=[]
    Gini_B_tot=[]
    Gini_B_interval_tot=[]
    Gini_interval=[]
    for time in range(simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time].sum())
        Gini_B_interval_tot.append(Gini_B_interval[time].sum())
        Gini.append(Gini_B_tot[time] + np.nansum(Gini_W[:,time]))
        Gini_interval.append(Gini_B_interval_tot[time] + Gini_W_interval[:, time].sum())




    return [Gini_JH,Gini_JJ,Gini_B,Gini_W,Gini_B_tot,Gini,Gini_JJ_interval,Gini_W_interval,Gini_B_interval,Gini_B_interval_tot,Gini_interval]

def calcola_gini_JJ_W( ns_at_time,bin_i,ns_tot_at_time,pop_tot,n_pezzi,pop_size,ns_pop_at_time, mu_IC,mu,pop,time):

    add_gini_w = []
    add_gini_JJ = []

    dif_abs_sum = 0
    dif_abs_sum_interval = 0
    for n in range(n_pezzi):
        for m in range(n_pezzi):
            #print("time", time)
            print("n", n)
            print("m", m)

            #print("n_pezzi", n_pezzi)

            dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[pop][bin_i[n]:bin_i[n + 1] , time],ns_at_time[pop][bin_i[m]:bin_i[m + 1] , time])))
            add_gini_JJ.append(dif_abs_aux.sum() / (2 * (pop_size[pop] ** 2) * mu_IC[pop][time]))
            dif_abs_sum = dif_abs_sum + dif_abs_aux.sum()
            if time == 0:
                ns_interval_n = ns_at_time[pop][bin_i[n]:bin_i[n + 1] , time]
                ns_interval_m = ns_at_time[pop][bin_i[m]:bin_i[m + 1] , time]
            else:
                ns_interval_n = ns_at_time[pop][bin_i[n]:bin_i[n + 1] , time] - ns_at_time[pop][bin_i[n]:bin_i[n + 1] , time - 1]
                ns_interval_m = ns_at_time[pop][bin_i[m]:bin_i[m + 1] , time] - ns_at_time[pop][bin_i[m]:bin_i[m + 1] , time - 1]
            dif_abs_sum_interval = dif_abs_sum_interval + np.abs(
                    np.subtract.outer(ns_interval_n, ns_interval_m)).sum()

    if time == 0:
        ns_pop_interval = ns_pop_at_time[pop][time]
        mu_interval = ns_tot_at_time[time] / pop_tot
    else:
        ns_pop_interval = ns_pop_at_time[pop][time] - ns_pop_at_time[pop][time - 1]
        mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

    mu_IC_interval = ns_pop_interval / pop_size[pop]
    Gini_JJ_interval=dif_abs_sum_interval / (2 * (pop_size[pop] ** 2) * mu_IC_interval)
    Gini_W_interval=mu_IC_interval * (pop_size[pop] ** 2) / ((pop_tot ** 2) * mu_interval * Gini_JJ_interval)
    Gini_JJ=np.sum(add_gini_JJ, 0)
    Gini_W=(mu_IC[pop][time] * (pop_size[pop] ** 2)) / ((pop_tot ** 2) * mu[time]) * Gini_JJ
    return [Gini_JJ_interval,Gini_W_interval,Gini_JJ,Gini_W]

def calcola_gini_JJ_W_new( ns_at_time,ns_tot_at_time,pop_tot,pop_size,ns_pop_at_time, mu_IC,mu,pop,time):

    add_gini_w = []
    add_gini_JJ = []

    dif_abs_sum = 0
    dif_abs_sum_interval = 0

    if time == 0:
        ns_pop_interval = ns_pop_at_time[pop][time]
        mu_interval = ns_tot_at_time[time] / pop_tot
    else:
        ns_pop_interval = ns_pop_at_time[pop][time] - ns_pop_at_time[pop][time - 1]
        mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

    mu_IC_interval = ns_pop_interval / pop_size[pop]
    aux2 = ns_at_time[pop][:, time]
    aux = ns_at_time[pop][:, time] != 0
    zeros = ns_at_time[pop][:, time] == 0
    n_zeros = zeros.sum()
    dif_abs_aux2 = np.abs(np.subtract.outer(aux2[aux], aux2[aux]))
    dif_abs_sum =dif_abs_aux2.sum()+2*aux2.sum()*n_zeros
    Gini_JJ= dif_abs_sum / (2 * (pop_size[pop] ** 2) * mu_IC[pop][time])
    if time == 0:
        ns_interval = ns_at_time[pop][:, time]
    else:
        ns_interval = ns_at_time[pop][:, time] - ns_at_time[pop][:, time - 1]
    aux2 = ns_interval
    aux = ns_interval != 0
    zeros = ns_interval == 0
    n_zeros = zeros.sum()
    dif_abs_aux2 = np.abs(np.subtract.outer(aux2[aux], aux2[aux]))
    dif_abs_sum = dif_abs_aux2.sum() + 2 * aux2.sum() * n_zeros
    dif_abs_sum_interval = dif_abs_sum
    Gini_JJ_interval = dif_abs_sum_interval / (2 * (pop_size[pop] ** 2) * mu_IC_interval)
    Gini_W_interval=mu_IC_interval * (pop_size[pop] ** 2) / ((pop_tot ** 2) * mu_interval * Gini_JJ_interval)

    Gini_W=(mu_IC[pop][time] * (pop_size[pop] ** 2)) / ((pop_tot ** 2) * mu[time]) * Gini_JJ
    return [Gini_JJ_interval,Gini_W_interval,Gini_JJ,Gini_W]

def calcola_gini_JJ_W_new2(ns_tot_at_time, pop_tot, pop_size, ns_pop_at_time, mu_IC, mu, pop, time):

    global ns_at_time
    add_gini_w = []
    add_gini_JJ = []

    dif_abs_sum = 0
    dif_abs_sum_interval = 0

    if time == 0:
        ns_pop_interval = ns_pop_at_time[pop][time]
        mu_interval = ns_tot_at_time[time] / pop_tot
    else:
        ns_pop_interval = ns_pop_at_time[pop][time] - ns_pop_at_time[pop][time - 1]
        mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

    mu_IC_interval = ns_pop_interval / pop_size[pop]
    aux2 = ns_at_time[pop][:, time]
    aux = ns_at_time[pop][:, time] != 0
    zeros = ns_at_time[pop][:, time] == 0
    n_zeros = zeros.sum()
    dif_abs_aux2 = np.abs(np.subtract.outer(aux2[aux], aux2[aux]))
    dif_abs_sum = dif_abs_aux2.sum() + 2 * aux2.sum() * n_zeros
    Gini_JJ = dif_abs_sum / (2 * (pop_size[pop] ** 2) * mu_IC[pop][time])
    if time == 0:
        ns_interval = ns_at_time[pop][:, time]
    else:
        ns_interval = ns_at_time[pop][:, time] - ns_at_time[pop][:, time - 1]
    aux2 = ns_interval
    aux = ns_interval != 0
    zeros = ns_interval == 0
    n_zeros = zeros.sum()
    dif_abs_aux2 = np.abs(np.subtract.outer(aux2[aux], aux2[aux]))
    dif_abs_sum = dif_abs_aux2.sum() + 2 * aux2.sum() * n_zeros
    dif_abs_sum_interval = dif_abs_sum
    Gini_JJ_interval = dif_abs_sum_interval / (2 * (pop_size[pop] ** 2) * mu_IC_interval)
    Gini_W_interval = mu_IC_interval * (pop_size[pop] ** 2) / ((pop_tot ** 2) * mu_interval * Gini_JJ_interval)

    Gini_W = (mu_IC[pop][time] * (pop_size[pop] ** 2)) / ((pop_tot ** 2) * mu[time]) * Gini_JJ
    return [Gini_JJ_interval, Gini_W_interval, Gini_JJ, Gini_W]

def calcola_JH(ns_at_time,bin_i,bin_j,ns_tot_at_time,pop_tot,n_pezzi,pop_size,ns_pop_at_time, mu_IC,i,j,time,Gini_JH,Gini_B_interval,Gini_JH_interval):

    add_GiniJH=[]
    dif_abs_sum = 0
    dif_abs_sum_interval=0
    for n in range(n_pezzi):
        for m in range(n_pezzi):

            dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1],time], ns_at_time[j][bin_j[m]:bin_j[m+1],time])))
            add_GiniJH.append(dif_abs_aux.sum())
            dif_abs_sum = dif_abs_sum + dif_abs_aux.sum()
            if time == 0:
                ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n + 1], time]
                ns_interval_j = ns_at_time[j][bin_j[m]:bin_j[m + 1], time]
            else:
                ns_interval_i = ns_at_time[i][bin_i[n]:bin_i[n+1],time] - ns_at_time[i][bin_i[n]:bin_i[n+1],time-1]
                ns_interval_j = ns_at_time[j][bin_j[m]:bin_j[m + 1] , time] - ns_at_time[j][bin_j[m]:bin_j[m + 1] , time - 1]
            dif_abs_sum_interval = dif_abs_sum_interval+np.abs(np.subtract.outer(ns_interval_i, ns_interval_j)).sum()

            if time == 0:
                ns_pop_interval_i = ns_pop_at_time[i][time]
                mu_interval = ns_tot_at_time[time] / pop_tot
                ns_pop_interval_j = ns_pop_at_time[j][time]

            else:
                ns_pop_interval_i = ns_pop_at_time[i][time] - ns_pop_at_time[i][time - 1]
                ns_pop_interval_j = ns_pop_at_time[j][time] - ns_pop_at_time[j][time - 1]
                mu_interval = (ns_tot_at_time[time] - ns_tot_at_time[time - 1]) / pop_tot

            mu_IC_interval_i = ns_pop_interval_i / pop_size[i]

            mu_IC_interval_j = ns_pop_interval_j / pop_size[j]

    Gini_JH_interval[time][i][j] = dif_abs_sum_interval / ((mu_IC_interval_i + mu_IC_interval_j) * pop_size[j] * pop_size[i])
    Gini_B_interval[time, i, j] = (mu_IC_interval_i + mu_IC_interval_j) * (pop_size[i] * pop_size[j]) / ((pop_tot ** 2) * mu_interval) * Gini_JH_interval[time][i][j]
    Gini_JH[time][i][j] = dif_abs_sum / ((mu_IC[i][time] + mu_IC[j][time]) * pop_size[j] * pop_size[i])

    return [Gini_JH_interval,Gini_B_interval,Gini_JH]
def compute_gini_WB_par(ns_at_time,ns_tot_at_time,ns_pop_at_time, pr_at_time,pr_at_time_IC,mu,mu_IC,spezzetta,n_pezzi,n,m):
    n_pop=len(ns_at_time)
    simulation_time_scaled = ns_tot_at_time.size
    dif_abs=[]
    pp_IC=[]
    Gini_W=[]
    Gini_JJ=[]
    Gini_B=np.zeros([simulation_time_scaled,n_pop,n_pop])



    for i in range(n_pop):
        pop_size=len(ns_at_time[i])


        bin_i = np.zeros(n_pezzi+1,int)
        for j in range(n_pezzi):
            bin_i[j]=int(j*int(pop_size / n_pezzi))
        bin_i[n_pezzi]=int(pop_size)

        for time in range(simulation_time_scaled):
            add_gini_w=[]
            add_gini_JJ=[]


            pp_IC_aux=((np.outer(pr_at_time_IC[i][bin_i[n]:bin_i[n+1]-1, time],pr_at_time_IC[i][bin_i[m]:bin_i[m+1]-1, time])))
            pp_aux=(np.outer(pr_at_time[i][bin_i[n]:bin_i[n+1]-1,time],pr_at_time[i][bin_i[m]:bin_i[m+1]-1,time]))
            dif_abs_aux=(np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1]-1,time], ns_at_time[i][bin_i[m]:bin_i[m+1]-1, time])))
            add_gini_w.append((pp_aux * dif_abs_aux).sum() / (2 * mu[time]))
            add_gini_JJ.append((pp_IC_aux * dif_abs_aux ).sum()/ (2 * mu_IC[i][time]))

            Gini_JJ.append(np.sum(add_gini_JJ,0))
            Gini_W.append(mu_IC[i][time]*(ns_pop_at_time[i][time]**2)/((ns_tot_at_time[time]**2)*mu[time])*Gini_JJ[-1])


    Gini_JH=np.zeros([simulation_time_scaled,n_pop,n_pop])

    for i in range(1,n_pop):
        for j in range(i):
            if (i!=j):



                aux_GB=[]
                add_GiniJH=[]
                if spezzetta:
                    pop_size=len(ns_at_time[i])
                    bin_i = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_i[n] = int(n * int(pop_size / n_pezzi))
                    bin_i[n_pezzi] = int(pop_size)

                    pop_size = len(ns_at_time[j])
                    bin_j = np.zeros(n_pezzi + 1, int)
                    for n in range(n_pezzi):
                        bin_j[n] = int(n * int(pop_size / n_pezzi))
                    bin_j[n_pezzi] = int(pop_size)
                    for time in range(simulation_time_scaled):
                        add_GiniJH=[]
                        pp_aux = (np.outer(pr_at_time_IC[i][bin_i[n]:bin_i[n+1]-1, time], pr_at_time_IC[j][bin_j[m]:bin_j[m+1]-1,time]))
                        dif_abs_aux = (np.abs(np.subtract.outer(ns_at_time[i][bin_i[n]:bin_i[n+1]-1,time], ns_at_time[j][bin_j[m]:bin_j[m+1]-1,time])))
                        add_GiniJH.append((pp_aux * dif_abs_aux).sum())
                        Gini_JH[time][i][j]=np.sum(add_GiniJH)/ (mu_IC[i][time] + mu_IC[j][time])
                else:
                    for time in range(simulation_time_scaled):
                        Gini_JH[time][i][j] = (np.outer(pr_at_time_IC[i][:, time],pr_at_time_IC[j][:, time]) * np.abs(np.subtract.outer(ns_at_time[i][:, time],ns_at_time[j][:, time]))).sum() / (mu_IC[i][time] + mu_IC[j][time])

                for time in range(simulation_time_scaled):
                    Gini_B[time,i,j]=(mu_IC[i][time]+mu_IC[j][time])*(ns_pop_at_time[i][time]*ns_pop_at_time[j][time])/((ns_tot_at_time[time]**2)*mu[time])*Gini_JH[time][i][j]



    Gini_W=np.array(Gini_W).reshape(n_pop,simulation_time_scaled)
    Gini_JJ = np.array(Gini_JJ).reshape(n_pop, simulation_time_scaled)

    Gini=[]
    Gini_B_tot=[]
    for time in range(simulation_time_scaled):
        Gini_B_tot.append(Gini_B[time].sum())
        Gini.append(Gini_B_tot[time] + Gini_W[:, time].sum())


    return [Gini_JH[1:simulation_time_scaled,:,:],Gini_JJ[:,1:simulation_time_scaled],Gini_B[1:simulation_time_scaled,:,:],Gini_W[:,1:simulation_time_scaled],Gini_B_tot[1:simulation_time_scaled],Gini[1:simulation_time_scaled]]



def compute_Theil_W_B(ns_at_time,ns_tot_at_time,ns_pop_at_time,mu,mu_IC):
    n_pop = len(ns_at_time)
    pop_size=np.zeros(n_pop)
    simulation_time_scaled = ns_tot_at_time.size
    s=np.zeros([n_pop,simulation_time_scaled])
    s_interval = np.zeros([n_pop, simulation_time_scaled])
    Theil_B_at_time=np.zeros([n_pop,simulation_time_scaled])
    Theil_for_pop_at_time=np.zeros([n_pop,simulation_time_scaled])
    Theil_B_tot_at_time=np.zeros(simulation_time_scaled)
    Theil_W_tot_at_time=np.zeros(simulation_time_scaled)

    Theil_B_at_time_interval = np.zeros([n_pop, simulation_time_scaled])
    Theil_for_pop_at_time_interval = np.zeros([n_pop, simulation_time_scaled])
    Theil_B_tot_at_time_interval = np.zeros(simulation_time_scaled)
    Theil_W_tot_at_time_interval = np.zeros(simulation_time_scaled)

    mu_interval=np.zeros(simulation_time_scaled)
    mu_IC_interval=np.zeros([n_pop, simulation_time_scaled])



    for i in range(n_pop):
        pop_size[i]=len(ns_at_time[i])
    pop_tot=pop_size.sum()

    for pop in range(n_pop):
        for time in range(1,simulation_time_scaled):
            Theil_for_pop_at_time[pop,time]=np.nan_to_num(compute_Theil(ns_at_time[pop][:, time], mu_IC[pop][time]),0)
            if time>0:
                mu_IC_interval[pop][time]=(mu_IC[pop][time] - mu_IC[pop][time - 1])
                mu_interval[time]=mu[time]-mu[time-1]
                Theil_for_pop_at_time_interval[pop, time] = compute_Theil(ns_at_time[pop][:, time]-ns_at_time[pop][:, time-1], mu_IC_interval[pop][time])
            else:

                mu_IC_interval[pop][time] = mu_IC[pop][time]
                mu_interval[time] = mu[time]
                Theil_for_pop_at_time_interval[pop, time] = compute_Theil(ns_at_time[pop][:, time], mu_IC_interval[pop][time])



    Theil_W_tot_at_time
    for pop in range(n_pop):
        s[pop,:]=np.nan_to_num(pop_size[pop]*mu_IC[pop][:]/(pop_tot*mu[:]))
        s_interval[pop, :] = np.nan_to_num(pop_size[pop] * mu_IC_interval[pop][:] / (pop_tot * mu_interval[:]))
        Theil_B_at_time[pop,:]=s[pop,:]*np.log(mu_IC[pop]/mu)
        Theil_B_at_time_interval[pop, :] = s_interval[pop, :] * np.log(mu_IC_interval[pop] / mu_interval)


    for time in range(1,simulation_time_scaled):
        Theil_B_tot_at_time[time]=Theil_B_at_time[np.isfinite(Theil_B_at_time[:,time]),time].sum()
        Theil_W_tot_at_time[time]=np.sum(Theil_for_pop_at_time[np.isfinite(Theil_for_pop_at_time[:,time]),time]*s[np.isfinite(Theil_for_pop_at_time[:,time]),time])#np.sum(np.isfinite(Theil_for_pop_at_time[:,time]*s[:,time]))
        Theil_B_tot_at_time_interval[time] = Theil_B_at_time_interval[np.isfinite(Theil_B_at_time_interval[:, time]), time].sum()
        Theil_W_tot_at_time_interval[time] = np.sum(Theil_for_pop_at_time_interval[np.isfinite(Theil_for_pop_at_time_interval[:,time]),time]*s_interval[np.isfinite(Theil_for_pop_at_time_interval[:,time]),time])#np.sum(np.isfinite(Theil_for_pop_at_time_interval[:, time] * s_interval[:, time]))


    Theil_at_time = Theil_W_tot_at_time+Theil_B_tot_at_time
    Theil_at_time_interval = Theil_W_tot_at_time_interval + Theil_B_tot_at_time_interval
#        mu_IC[pop][time]
 #       ns_at_time[pop][:, time]


    return  [Theil_for_pop_at_time,Theil_B_at_time,Theil_B_tot_at_time,Theil_W_tot_at_time,Theil_at_time,Theil_for_pop_at_time_interval,Theil_B_at_time_interval,Theil_B_tot_at_time_interval,Theil_W_tot_at_time_interval,Theil_at_time_interval,s,s_interval]
def compute_Theil(ns,mu):

    dim_pop=ns.shape[0]

    return np.sum(np.nan_to_num(ns*np.log(ns/mu))/(dim_pop*mu))