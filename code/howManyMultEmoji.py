import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
from keras.models import Model
from keras.layers import *
from keras.preprocessing import sequence as ksq
from scipy import spatial
#path = '2016-11-09 00:04:21.211531 600000.h5'
path = '2016-11-09 22:20:43.533474 600000.h5'

def cosine_sim(y_pred, y_true):
    return 1 - spatial.distance.cosine(y_pred, y_true)

fromI =0 
toI =600000
data_num = toI-fromI
maxlen = 32
embed_dim = 300

print('==get data==')
gtf = open("../data/data1108/tweetEmoji.csv")#ground truth



print('==making ground truth label==')
gtVectors=[]
count1,count2=0,0
for i,l in enumerate(gtf):
	if i >= toI or i < fromI:
		continue

	line=l.rstrip().split(',')
        #gtVectors.append([int(x) for x in line[2].split()])
        if len(line[2].split()) ==1:
            count1 +=1
        elif len(line[2].split())>1:
            count2 +=1

print('==get data done==')
#################################
print count1
print count2





