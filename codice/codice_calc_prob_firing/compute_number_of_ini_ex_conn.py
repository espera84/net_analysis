
import numpy as np
import h5py
from joblib import Parallel, delayed
import time
import os
import pickle
import matplotlib.pyplot as plt



def comp_ex_n_con(f_con,l_con,n_SP_PC):
    f_pyr = h5py.File(filename_PC, "r")
    pyr_connection_list = list(f_pyr.keys())
    n_con_ex_on_pyr2 = np.zeros(n_SP_PC + 1)
    np.add.at(n_con_ex_on_pyr2, f_pyr[pyr_connection_list[0]][f_con:l_con, 1], 1)

    return n_con_ex_on_pyr2

def comp_in_n_con(i,n_worker,n_SP_PC):


    f_in = h5py.File(filename_in, "r")
    in_connection_list = list(f_in.keys())
    n_con_in_on_pyr = np.zeros(n_SP_PC+1)
    for connection_type in in_connection_list:
        #print(connection_type)
        if ('to_SP_PC') in connection_type:
            n_con = f_in[connection_type][:, 0].shape[0]
            bin_size = int(np.ceil(n_con / n_workers))
            f_con=i*bin_size
            l_con=min((i+1)*bin_size,n_con)
            np.add.at(n_con_in_on_pyr, f_in[connection_type][f_con:l_con, 1].astype(int), 1)

    return n_con_in_on_pyr




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

N_SP_PC=f_pos['SP_PC'][:,0].shape[0]
N_AA=f_pos['AA'][:,0].shape[0]
N_BP=f_pos['BP'][:,0].shape[0]
N_BS=f_pos['BS'][:,0].shape[0]
N_CCKBC=f_pos['CCKBC'][:,0].shape[0]
N_IVY=f_pos['IVY'][:,0].shape[0]
N_NGF=f_pos['NGF'][:,0].shape[0]
N_OLM=f_pos['OLM'][:,0].shape[0]
N_PPA=f_pos['PPA'][:,0].shape[0]
N_PVBC=f_pos['PVBC'][:,0].shape[0]
N_SCA=f_pos['SCA'][:,0].shape[0]
N_TRI=f_pos['TRI'][:,0].shape[0]

n_con_ex_on_pyr=np.zeros(N_SP_PC+1)
n_con=f_pyr[pyr_connection_list[0]][:,1].shape[0]#10000#


n_workers=os.cpu_count()
print(n_workers)
bin_size=int(np.ceil(n_con/n_workers))

t1 = time.time()
Ltt = Parallel(n_jobs=n_workers, verbose=50)(delayed(comp_ex_n_con)(i*bin_size,min(n_con,(i+1)*bin_size),N_SP_PC) for i in range(n_workers))
elapsed = time.time() - t1


for i in range(n_workers):
    n_con_ex_on_pyr=n_con_ex_on_pyr+Ltt[i]

t2 = time.time()
Ltt = Parallel(n_jobs=n_workers, verbose=50)(delayed(comp_in_n_con)(i,n_workers,N_SP_PC) for i in range(n_workers))
elapsed2 = time.time() - t2

n_con_in_on_pyr=0
for i in range(n_workers):
    n_con_in_on_pyr=n_con_in_on_pyr+Ltt[i]
#elapsed = time.time() - t

id=np.arange(n_con_ex_on_pyr.shape[0])
with open('Pyr_connection_info.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([id[1:],n_con_ex_on_pyr[1:],n_con_in_on_pyr[1:]], f)


plt.figure();
plt.bar(range(n_con_ex_on_pyr.shape[0]),n_con_ex_on_pyr, label="ex synapses" )
plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
plt.xlabel('neuron ID')
plt.ylabel('number of ex input synapses ')
plt.savefig('neuron_ID_ex_connection.png')


plt.figure();
plt.bar(range(n_con_in_on_pyr.shape[0]),n_con_in_on_pyr, label="in synapses" )
plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
plt.xlabel('neuron ID')
plt.ylabel('number of in input synapses ')
plt.savefig('neuron_ID_in_connection.png')

n_con_ex_max=n_con_ex_on_pyr.max()
n_with_n_ex_con=np.zeros(int(n_con_ex_max))
for i in range(int(n_con_ex_max)):
    n_with_n_ex_con[i]=np.sum(n_con_ex_on_pyr==i)

n_con_in_max=n_con_in_on_pyr.max()
n_with_n_in_con=np.zeros(int(n_con_in_max))
for i in range(int(n_con_in_max)):
    n_with_n_in_con[i]=np.sum(n_con_in_on_pyr==i)

plt.figure();
plt.bar(range(n_with_n_ex_con.shape[0]),n_with_n_ex_con, label="ex synapses" )
plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
plt.xlabel('n of ex con')
plt.ylabel('number of neuron ')
plt.savefig('number_of_neuron_ex_connection.png')


plt.figure();
plt.bar(range(n_with_n_in_con.shape[0]),n_with_n_in_con, label="in synapses" )
plt.legend(bbox_to_anchor=(0.2, 0.2), loc='upper left', borderaxespad=0.)
plt.xlabel('n of in con')
plt.ylabel('number of neuron ')
plt.savefig('number_of_neuron_in_connection.png')
