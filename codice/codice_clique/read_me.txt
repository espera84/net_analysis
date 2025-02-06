input:
all'interno del file clique_configuration.json è possibile definire:

-intervallo della sim da analizzare tramite :
	"start_time" e "end_time"
-dimensione del bin da utilizzare per binarizzare (neurone attivo/non attivo) l'attività di ogni neurone ("bins_dimension")
-nome della cartella contenente il file di attività della sim:"folder_simulation_name"
-percentuale di bin in cui il neurone deve produrre spikes affinchè il neurone venga preso in considerazione nell'analisi:"percentual_of_firing_bins_for_active"
-percentuale massima di differenza tra le attività binarizzate di 2 neuroni affinchè le attività dei 2 neuroni si considerino correlate "percentual_of_different_bins_for_correlation"
-numero di traslazioni tra i vettori binari da testare per verificare la distanza minima ("n_shift") (da testare meglio il funzionamento)
-numero massimo di neuroni su cercare i cliques: "n_of_neurons_max_to_select":288027 (da diminuire in caso di eccessivo peso computazionale)
- dimensione minima delle componenti connesse da prendere in considerazione: "n_of_neurons_min_for_connected_component"
- dimensione minima dei clique da prendere in considerazione:	"n_of_neurons_min_for_clique"
- numerocomplessivo di neuroni della rete analizzata : "n_of_neurons_in_the_network"

dalla cartella "/input_data/data_net/" prende in input:
	-il file positions.hdf5 contenente posizioni e id di tutti i neuroni 


output:

nella cartella result_path="results/nome_sim/clique_interval_"+str(t_initial_analysis)+"_"+str(t_final_analysis)+"interval_dim_"+str(interval_dim)+"/"" salva:
- neurons_spiking.pkl contenente 3 variabili:
	-is_spking: np.array booleano dim:(n_neuroni) 
	-fn_tutti: np.array contenente in posizione "i" una lista con tutti i neuroni che sparano durante l'i-esimo intervallo temporale
	-ns_fn_tutti: np.array contenente in posizione "i" il numero di spikes per cogni neurone presente nella i-sima posizione di fn_tutti
- data_neurons_distances'+str(soglia_attivi)+'_n_neurons'+str(n_neuroni_attivi)+'.pkl' contenente 3 variabili:
	- n_neuroni_attivi
	-neuroni_attivi: np.array con gli id dei neuroni attivi
	-dif: matrice di dimensione(n_neuroni_attivi,n_neuroni_attivi) contenente in posizione (i,j) la distanza di hamming tra i vettori binari associati all'attività di firing dei neuroni neuroni_attivi[i] e neuroni_attivi[j]  

nella cartella result_sub_path=result_path+"soglia_attivi_" + str(perc_attivi)+"soglia_corr"+str(perc_corr)+"n_shift"+str(n_shift)+"_cl/" salva:

- data_cc.pkl contenente 6 variabili:
	- mat_di_corr: matrice booleana (abbiamo true in posizione (i,j) se dif(i,j) è minore della soglia di correlazione
	- neurons_correlated: matrice di dimensione( 2, numero di coppie di neuroni con firing correlato) che contiene le coppie di neuroni correlati (ex: se è presente la coppia i, j il neuroni_attivi[i] e il neuroni_attivi[j] sono correlati)
	- id_comp_connesse: lista degli id associati alle componenti connesse del grafo costruito dalla mat_di_corr (con almeno "n_of_neurons_min_for_connected_component" nodi)
	- n_comp_connesse
	- list_of_cliques_cc: lista che contiene in posizione "i" i cliques presenti all'interno della i-sima componente connessa
	- labels: np.array di dimensione (n_attivi) che in posizione i contiene l'id della componente connessa a cui appartiene il neurone neuroni_attivi[i]

- clique_info.pkl contenente 6 variabili:
	- id_neu_cliques: lista per ogni comp connessa dei neuroni associati ai vari clique (ex id_neu_cliques[i][j] contiene gli id dei neuroni del j-simo clique della componente connessa i)
	- indici_cl: lista degli indici dei clique
	- cl_spk: vettore di dimensione (2, numero spikes di tutti i clique) che contiene le coppie (tempo di spikes, id_neurone_alternativo) (id_neurone_alternativo non corrisponde all'id del neurone nella rete)
	- col: vettore di dimensione (numero spikes di tutti i clique) contine in posizione "i" l'id del clique in cui si trova il neurone cl_spk[1,i] (VERIFICA)
	- Isi_cliques: lista degli isi di ogni clique  Isi_cliques[i] contiene gli isi dell'i-simo clique 
	- perc_pyr: vettore con le percentuali di piramidali per ogni clique
	
