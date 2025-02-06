input:
all'interno del file sim_configuration.json è possibile definire:
-intervallo della sim da analizzare tramite :
	"start_time" e "end_time"
-dimensione del bin da utilizzare per i grafici a barre ()
-nome della cartella contenente il file di attività della sim:"folder_simulation_name"

dalla cartella "/input_data/data_net/" prende in input:
	-i files conteneti id e neuroni delle diverse slices
	-il file positions.hdf5 contenente posizioni e id di tutti i neuroni 


output:
per ogni slice trasversale dalla 0 all 18 calcola e produce le immagini (salvate nella cartella "results/nome_sim/"hist_slices/") di:
-gli istogrammi del firing
-gli istogrammi del firing normalizzati (sommatoria dell'istogramma=1)
-grafico a barre del numero di neuroni che sparano per ogni input
-grafico a barre della percentuale di neuroni che sparano per ogni input 
-produce inoltre un file di visualizzazione delle slice sull'intera struttura
