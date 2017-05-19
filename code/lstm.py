import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["THEANO_FLAGS"]='mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0'

import numpy as np
from keras.models import Model
from keras.layers import *
import keras.optimizers
from keras.preprocessing import sequence as ksq
from keras.callbacks import ModelCheckpoint
from theano import tensor as T
import datetime
from scipy import spatial
import time
import theano
start_time = time.time()

#x: training data -> (#data, maxlen of a sentence by words, embed_dim of w2v)
#[case 1. word2vec approach] y: label -> (#data, embed_dim)
#[case 2. one-hot encoding approach] y: label -> (#data, #label)

#data_path= 'animation'
data_path = 'demo'

#data_num = 589
data_num = 11
#maxlen = 525
maxlen = 298
embed_dim = 3#300
nb_epoch = 3000
batch_size = 256


def cosine_sim(y_pred, y_true):
    pred_norm = T.sum(T.square(y_pred), axis=1, keepdims=True)
    pred_norm = T.switch(pred_norm <= 0, 1.0, pred_norm)
    y_pred = y_pred / T.sqrt(pred_norm)
    true_norm = T.sum(T.square(y_true), axis=1, keepdims=True)
    true_norm = T.switch(true_norm <= 0, 1.0, true_norm)
    y_true = y_true / T.sqrt(true_norm)
    return -K.mean(K.sum((y_true * y_pred), axis=1))

def getData(maxSize):
    print('==get data==')
    '''
    bpsf = open("../data/"+data_path+'/bps.txt')#sentence
    r_pf= open("../data/"+data_path+'/r_p.txt')#sentence
    s_pf= open("../data/"+data_path+'/s_p.txt')#sentence
    '''
    bpsf = open("../data/"+data_path+'/bps_result.txt')#sentence
    r_pf= open("../data/"+data_path+'/receive_count_result.txt')#sentence
    s_pf= open("../data/"+data_path+'/send_count_result.txt')#sentence

    inputVectors = []
    labelVectors = []
        
    dummy = [0]*(embed_dim)

    for l in bpsf:
        line = l.rstrip().split(',')
        
        lmax = max([float(x) for x in line[1:]])
        #inputV = [[float(x)/lmax]+dummy for x in line[1:]]
        inputV = [[float(x)/lmax] for x in line[1:]]

        labelV = []
        if line[0] == "0":
            labelV = [1,0,0]#tcp

        elif line[0] == "1":
            labelV = [0,1,0]#udp

        elif line[0] == "2":
            labelV = [0,0,1]#bad tcp


        labelV = labelV+dummy[3:]
        inputVectors.append(inputV)
        labelVectors.append(labelV)

    for i,l in enumerate(s_pf):
        line = l.rstrip().split(',')
        
        lmax = max([float(x) for x in line[1:]])
        for j,x in enumerate(inputVectors[i]):
            x.append(float(line[j+1])/lmax)



    for i,l in enumerate(r_pf):
        line = l.rstrip().split(',')
       
        lmax = max([float(x) for x in line[1:]])
        for j,x in enumerate(inputVectors[i]):
            if lmax == 0:
                x.append(0.0)
            else:
                x.append(float(line[j+1])/lmax)
            

    #dummy
    for i in range(len(inputVectors)):
        for j in range(len(inputVectors[i])):
            for k in range(embed_dim-3):
                inputVectors[i][j].append(float(0))

    print inputVectors[0]
         


    #print inputVectors
    #print labelVectors
    return inputVectors, labelVectors

print('==copying array==')


x_d, y_d = getData(data_num)
x_d = np.array(x_d)#[:,:,np.newaxis]#3d
y = np.array(y_d)

print x_d.shape
print y.shape

print("--- %s seconds ---" % (time.time() - start_time))
print('==padding==')

x = ksq.pad_sequences(x_d, maxlen=maxlen, dtype='float32')
#x = x[:,:,np.newaxis]
print x.shape
print("--- %s seconds ---" % (time.time() - start_time))
print('==Model initialization==')

input_x = Input(shape=(maxlen, embed_dim), dtype='float32')
xr = LSTM(embed_dim, return_sequences=False)(input_x)#####


model = Model(input=[input_x], output=[xr])

optim = keras.optimizers.Adam(lr=0.001)
model.compile(loss={'lstm_1': 'mse'}, optimizer=optim, metrics=[], loss_weights={'lstm_1': 1.0})
#model.compile(loss={'lstm_1': cosine_sim}, optimizer=optim, metrics=[], loss_weights={'lstm_1': 1.0})

model.summary()

weight_file = '..//weight//'+str(datetime.datetime.now())+' '+str(data_num)+'.h5'

print("--- %s seconds ---" % (time.time() - start_time))




#one-cue 
checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_weights_only=True, save_best_only=True)
model.fit([x], y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.15, verbose=1, callbacks=[checkpointer])

print("--- %s seconds ---" % (time.time() - start_time))

