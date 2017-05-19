
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
inf = open('./test_input.txt','w')
outf = open('./test_result.txt', 'w')




def cosine_sim(y_pred, y_true):
    return 1 - spatial.distance.cosine(y_pred, y_true)


words = sys.argv[1:]
print words


#print('==loading w2v model==')
#w2v = word2vec.Word2Vec.load_word2vec_format(vpath, binary=True)

print('==get vectors from model==')
''' #from w2v 
sent=[]
for w in words:
    sent.append(w2v[w]);
sent = np.array(sent)
print sent.shape
sentsVectors = []
sentsVectors.append(sent)
sentsVectors = np.array(sentsVectors)
'''


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

sents = []
for w in words:
    for dic in wordDic:
        if w == dic[0]:
            print 'match:',w
            sents.append(dic[2:])
for s in sents:
    inf.write(str(s)+'\n')
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
print x

model.load_weights('..//weight//'+wpath)
print('predicting...')
output_y = model.predict([x], 256)#test x has similar dim to x, but diff # of data


'''
print x
print len(x[0])

print 'y',len(output_y)
print 'y',len(output_y[0])
'''
outf.write(str(output_y[0]))
'''
for y in output_y[0]:
    outf.write(str(y)+', ')
'''
print('==making emoji dic==')

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

predictResult = []
for i,y in enumerate(output_y):
    line = []
    for e in emojiDic:
    	dist = cosine_sim(y, np.array(e[2:]))#metric
	label = e[0]
        line.append((label,dist))
    predictResult.append(sorted(line,key=lambda tup: tup[1],reverse = True))#[:joy:,dist]

for x in predictResult:
    for y in x:
        print y

print("--- %s seconds ---" % (time.time() - start_time))
