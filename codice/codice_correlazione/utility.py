import numpy as np
def unify_families(mat_contents,pop,start_time,simulation_time,without_zeros):
    list_of_Matrix_spk = []
    n_pop = pop.__len__()
    for i in range(n_pop):
        empty = True
        for j in pop[i]:
            aux = mat_contents['spikeData'][0][j][0][1].A[:, range(start_time, len(
                mat_contents['spikeData'][0][j][0][1].A[1,
                :]))]  # mat_contents['spikeData'][0][0][0][1].A[:,range(2,18,5)]
            if without_zeros:
                aux = aux[aux.sum(1) != 0, :]
            if (simulation_time > aux.shape[1]):
                aux = np.concatenate((aux, np.zeros([aux.shape[0], simulation_time - aux.shape[1]])), axis=1)
            else:
                aux = aux[:, 0: simulation_time]
            if (empty):
                sin_class = aux
                empty = False
            else:
                sin_class = np.concatenate((sin_class, aux), axis=0)

        list_of_Matrix_spk.append(sin_class)

    return list_of_Matrix_spk

def bin_selection(spk_list,start_time,simulation_time):

    [hist, bins] = np.histogram(
        spk_list[np.logical_and(spk_list[:, 1] > start_time, spk_list[:, 1] < (start_time + simulation_time)), 1],
        int(np.floor(simulation_time / 100)))
    soglia = (hist.mean() + hist.min()) * 2 / 3
    hist_sub_soglia = hist > soglia
    pos_hist_sub_sog = np.where(hist_sub_soglia == False)
    vet_aux = pos_hist_sub_sog[0][:-1] == pos_hist_sub_sog[0][1:] - 1
    in_interval = False
    first_element = False
    init_intervals = []
    end_intervals = []
    pos_init_intervals = []
    pos_end_intervals = []
    for i in range(vet_aux.__len__()):
        if not in_interval:
            if vet_aux[i] == True:
                in_interval = True
                first_element = True
                init_intervals.append(bins[pos_hist_sub_sog[0][i]])
                end_intervals.append(bins[pos_hist_sub_sog[0][i]])
                pos_init_intervals.append(pos_hist_sub_sog[0][i])
                pos_end_intervals.append(pos_hist_sub_sog[0][i])
        else:
            if vet_aux[i] == True:
                if vet_aux[i - 1] == True:
                    end_intervals[-1] = bins[pos_hist_sub_sog[0][i + 1]]
                    pos_end_intervals[-1] = pos_hist_sub_sog[0][i + 1]
            else:
                in_interval = False

    pos_minimi = []
    for i in range(pos_end_intervals.__len__()):
        pos_minimi.append(pos_init_intervals[i] + hist[pos_init_intervals[i]:pos_end_intervals[i] + 1].argmin())
    if pos_minimi[0] != 0:
        pos_minimi = [0] + pos_minimi
    if pos_minimi[0] != 0:
        pos_minimi = [0] + pos_minimi
    if pos_minimi[-1] != hist.__len__() - 1:
        pos_minimi.append(hist.__len__() - 1)
    intervals = bins[pos_minimi]
    n_intervals_points = intervals.__len__()

    return [intervals,n_intervals_points]