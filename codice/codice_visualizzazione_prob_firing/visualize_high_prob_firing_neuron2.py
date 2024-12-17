import numpy as np
from extract_spike_num_and_freq import features_extraction_different_format_red
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import h5py
import csv
import pandas as pd

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

work_path="C:/Users/emili/Desktop/CNR/2023-24/figure4ae_data_code/codice_python/neuroni_alta_attivabilita/"

points = go.Scatter3d(x=f_pos[pos_neuron_list[10]][:, 1],
                      y=f_pos[pos_neuron_list[10]][:, 2],
                      z=f_pos[pos_neuron_list[10]][:, 3],
                      mode='markers',
                      marker=dict(size=1,
                                  color='gray',
                                  showscale=True, opacity=0.1)
                      ,

                      )

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0)
                   )
fig2 = go.Figure(data=points, layout=layout)

# lightsteelblue, lightyellow, lime, limegreen,
# linen, magenta, maroon, mediumaquamarine,
# mediumblue, mediumorchid, mediumpurple,
# mediumseagreen, mediumslateblue, mediumspringgreen,
# mediumturquoise, mediumvioletred, midnightblue,
# mintcream, mistyrose, moccasin, navajowhite, navy,
# oldlace, olive, olivedrab, orange, orangered,
# orchid, palegoldenrod, palegreen, paleturquoise,
# palevioletred, papayawhip, peachpuff, peru, pink,
# plum, powderblue, purple, red, rosybrown,
# royalblue, saddlebrown, salmon, sandybrown,
# seagreen, seashell, sienna, silver, skyblue,
# slateblue, slategray, slategrey, snow, springgreen,
# steelblue, tan, teal, thistle, tomato, turquoise,
# violet, wheat, white, whitesmoke, yellow,yellowgreen

color_discrete_sequence = ["orange","red","red", "green","green", "blue", "blue", "pink", "pink","yellow","yellow","olive","springgreen","purple","moccasin","orange"]
for n in range(1,10):

    list_of_neuron = pd.read_csv(work_path+"altaAttivabilitaDisgiunti"+str(n)+"_id.csv", header=None)
    #spk_list=np.array(list_of_list_spk)
    aux=np.array(list_of_neuron[0])
    posizioni=[]
    for i in aux:
        #print(i)
        posizioni.append(f_pos[pos_neuron_list[10]][i-1,1:])

    fig2.add_scatter3d(x = np.array(posizioni)[:,0],
                           y = np.array(posizioni)[:,1],
                           z = np.array(posizioni)[:,2],
                           mode = 'markers',
                           marker = dict( size = 1,
                                          color=color_discrete_sequence[n],
                                          showscale=True)
                      )  # ,showscale=True), )




fig2.write_html(work_path +"scatter_all_2.html")

