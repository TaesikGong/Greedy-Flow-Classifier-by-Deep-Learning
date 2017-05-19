
#!/usr/bin/python
import os
os.environ["THEANO_FLAGS"]='mode=FAST_RUN,floatX=float32,device=cpu,lib.cnmem=0'

from keras.models import Model
from keras.layers import *
from gensim.models import word2vec
from keras.preprocessing import sequence as ksq
import sys
from scipy import spatial
import time
start_time = time.time()

data_dir = '11-22_no_under_10_words_good_index'
wpath = '2016-11-22 15:41:01.015735 780000.h5'
vpath ='../GoogleNews-vectors-negative300.bin'

maxlen = 10
embed_dim = 300


emof = open("../data/"+data_dir+"/emojiVectors.csv")#emoji
worf = open("../data/"+data_dir+"/wordVectors.csv")#word
linkf = open("./emojiLink.csv")#link
inf = open('./test_input.txt','r')
outf = open('./test_result.csv', 'w')




def cosine_sim(y_pred, y_true):
    return 1 - spatial.distance.cosine(y_pred, y_true)

print('==making dic==')

emojiDic=[]
for l in emof:
	line = []
	for i,x in enumerate(l.rstrip().split(',')):
		if i == 0:
			line.append(x)
		elif i == 1:
			line.append(int(x))
		else :
			line.append(float(x))
	emojiDic.append(line)

wordDic=[]
for l in worf:
	line = []
	for i,x in enumerate(l.rstrip().split(',')):
		if i == 0:
			line.append(x)
		elif i == 1:
			line.append(int(x))
		else :
			line.append(float(x))
	wordDic.append(line)
linkDic = {}
for l in linkf:
    line = l.rstrip().split(',')    
    
    linkDic[line[0]]=line[2]


words = sys.argv[1:]
print words

for i,l in enumerate(inf):
    print('processing:'+str(i+1))
    words = []
    words = l.rstrip().split()


    print('==get vectors from model==')
    sents = []
    matched = []
    for w in words:
        for dic in wordDic:

           wt =  w.replace('!','').replace('.','').replace('?','').replace("'",'').replace('"','').lower()
           if wt == dic[0]:
                matched.append(wt)
                sents.append(dic[2:])
    print ('matched:'+str(matched))
    sentsVectors = []
    sentsVectors.append(np.array(sents))
    
    sentsVectors = np.array(sentsVectors)
    
    
    print('==LSTM starts==')
    input_x = Input(shape=(maxlen, embed_dim), dtype='float32')
    xr = LSTM(embed_dim,return_sequences=False)(input_x)
    
    model = Model(input=[input_x], output=[xr])
    
    
    print('copying x...')
    x_d = sentsVectors
    x = ksq.pad_sequences(x_d, maxlen=maxlen, dtype='float32')
    
    model.load_weights('..//weight//'+wpath)
    print('predicting...')
    output_y = model.predict([x], 256)#test x has similar dim to x, but diff # of data
    
    
    predictResult = []
    for i,y in enumerate(output_y):
        line = []
        for e in emojiDic:
            dist = cosine_sim(y, np.array(e[2:]))#metric
            label = e[0]
            line.append((label,dist))
        predictResult.append(sorted(line,key=lambda tup: tup[1],reverse = True))#[:joy:,dist]
    
    for x in predictResult:
        for i,y in enumerate(x):#y[1] = rank
            outf.write(l.rstrip()+','+str(y[0])+','+str(i+1)+','+str(linkDic[y[0]])+'\n')
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
