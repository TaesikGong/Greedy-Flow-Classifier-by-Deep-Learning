import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"]='mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0'

import h5py
import keras
from keras.models import Model
from keras.layers import *
from keras.preprocessing import sequence as ksq
import numpy as np
f = '2016-11-22 15:41:01.015735 780000.h5'
maxlen = 10
savePath = '../weight/txt/2/'
wpath = '..//weight//'+f

embed_dim = 300
input_x = Input(shape=(maxlen, embed_dim), dtype='float32')
xr = LSTM(embed_dim,return_sequences=False)(input_x)

model = Model(input=[input_x], output=[xr])
model.load_weights(wpath)

layers = []
f = h5py.File(wpath,'r')
for i, key in enumerate(f.keys()[1:]):
    #print i,key
    layer_weights = {}
    for value in f[key].values():
        layer_weights[value.name.split('/')[-1]] = np.array(value)
        print '#############',value.name, np.array(value).shape
    layers.append(layer_weights)

#print layers                                  

for l in model.layers:
    print '#',l.trainable_weights

for i, l in enumerate(layers):
    for key in l.keys():
        np.savetxt(savePath + str(i) + '_' + key +'.txt', l[key])
'''
print h5py.File(path)
print h5py.File(path).keys()
print h5py.File(path)['input_1']
print h5py.File(path)['lstm_1']
'''
