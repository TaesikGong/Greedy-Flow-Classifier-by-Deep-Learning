import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"]='mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0'


import numpy as np
from keras.models import Model
from keras.layers import *
from keras.preprocessing import sequence as ksq
from scipy import spatial
import time
import random 
start_time = time.time()

#data_path = 'animation'
data_path = 'real'

#wpath = '2016-12-10 04:54:13.609954 660.h5'#mse
wpath = '2016-12-12 01:55:42.445220 1680.h5'#cossim




fromIdx = 0
#toIdx = 589- 1
toIdx = 28- 1

data_num = toIdx - fromIdx + 1

#maxlen =525
maxlen = 293
embed_dim = 3



def cosine_sim(y_pred, y_true):
    return 1 - spatial.distance.cosine(y_pred, y_true)
def mse(y_pred, y_true):
    return ((y_pred - y_true)**2).mean(axis=0)

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


for i,l in enumerate(bpsf):
    if i<fromIdx or i>toIdx:
        continue

    line = l.rstrip().split(',')
    
    lmax = max([float(x) for x in line[1:]])
    lmax = max(1,lmax)
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

    if i<fromIdx or i>toIdx:
        continue


    line = l.rstrip().split(',')
    
    lmax = max([float(x) for x in line[1:]])
    lmax = max(1,lmax)
    for j,x in enumerate(inputVectors[i-fromIdx]):
        x.append(float(line[j+1])/lmax)



for i,l in enumerate(r_pf):
    if i<fromIdx or i>toIdx:
        continue


    line = l.rstrip().split(',')
   
    lmax = max([float(x) for x in line[1:]])
    lmax = max(1,lmax)
    for j,x in enumerate(inputVectors[i-fromIdx]):
        if lmax == 0:
            x.append(0.0)
        else:
            x.append(float(line[j+1])/lmax)

#dummy
for i in range(len(inputVectors)):
    for j in range(len(inputVectors[i])):
        for k in range(embed_dim-3):
            inputVectors[i][j].append(float(0))

         

tcp = [1,0,0]+dummy[3:]
udp = [0,1,0]+dummy[3:]
btcp= [0,0,1]+dummy[3:]

print tcp
print udp
print btcp 
print("--- %s seconds ---" % (time.time() - start_time))

print('==get data done==')


input_x = Input(shape=(maxlen, embed_dim), dtype='float32')
xr = LSTM(embed_dim,return_sequences=False)(input_x)

model = Model(input=[input_x], output=[xr])
model.load_weights('..//weight//'+wpath)


for i in range(maxlen):
    print('copying x... i:',i)
    x_d =  [x[:i+1] for x in inputVectors]
    x = ksq.pad_sequences(x_d, maxlen=maxlen, dtype='float32')
    
    #print('predicting y...')
    output_y = model.predict([x], 256)#test x has similar dim to x, but diff # of data
    predictResult=[]
    
     
    for i,y in enumerate(output_y): 
        #if i%(data_num/10)== 0:
            #print("--- %s seconds ---" % (time.time() - start_time))
            #print 'comparing...',i,'/',data_num
            #pass
        tcpMSE = [mse(y, np.array(tcp))]
        udpMSE = [mse(y, np.array(udp))]
        btcpMSE = [mse(y, np.array(btcp))]
        '''
        tcpMSE = [-cosine_sim(y, np.array(tcp))]
        udpMSE = [-cosine_sim(y, np.array(udp))]
        btcpMSE = [-cosine_sim(y, np.array(btcp))]
        '''



        label = 'tcp'
        result = tcp
        MSE = tcpMSE
        '''
        if tcpMSE > btcpMSE:
            result = btcp 
            MSE = btcpMSE 
            label = 'btcp'
        '''
        '''
        if tcpMSE > udpMSE:
            result = udp
            MSE = udpMSE 
            label = 'udp'

        '''
#three
        if tcpMSE > udpMSE:
            if udpMSE > btcpMSE:
                result = btcp 
                MSE = btcpMSE 
                label = 'btcp'
            else:
                result = udp
                MSE = udpMSE
                label = 'udp'
        else:
            if tcpMSE > btcpMSE:
                result = btcp 
                MSE = btcpMSE 
                label = 'btcp'
            else:
                result = tcp 
                MSE = tcpMSE 
                label = 'tcp'
        print (label,tcpMSE,udpMSE,btcpMSE,labelVectors[i])
        predictResult.append([result,MSE])
       
    #ground truth
    rm = [[0,0,0],[0,0,0],[0,0,0]]#result matrix

    for i,x in enumerate(labelVectors):
        p=predictResult[i][0]
        if x==tcp:
            if p == tcp:
                rm[0][0]+=1
            elif p == udp:
                rm[0][1]+=1
            elif p == btcp:
                rm[0][2]+=1

        elif x==udp:
            if p == tcp:
                rm[1][0]+=1
            elif p == udp:
                rm[1][1]+=1
            elif p == btcp:
                rm[1][2]+=1
        elif x==btcp:
            if p == tcp:
                rm[2][0]+=1
            elif p == udp:
                rm[2][1]+=1
            elif p == btcp:
                rm[2][2]+=1

    print rm[0][0],rm[0][1],rm[0][2] 
    print rm[1][0],rm[1][1],rm[1][2] 
    print rm[2][0],rm[2][1],rm[2][2] 















    
    g_tcp_count=0.0
    g_udp_count=0.0
    g_btcp_count=0.0

    for i,x in enumerate(labelVectors):
        if x == tcp:
            g_tcp_count+=1
        elif x == udp:
            g_udp_count+=1
        elif x == btcp:
            g_btcp_count+=1

    #predicted
    total_count=0.0
    tcp_count=0.0
    udp_count=0.0
    btcp_count=0.0
    #wrongly predicted
    w_total_count=0.0
    w_tcp_count=0.0
    w_udp_count=0.0
    w_btcp_count=0.0


    for i,x in enumerate(predictResult):
        if x[0] == labelVectors[i]:
            total_count+=1
            if x[0] == tcp:
                tcp_count+=1
            elif x[0] == udp:
                udp_count+=1
            elif x[0] == btcp:
                btcp_count+=1
        else:
            w_total_count+=1
            if x[0] == tcp:
                w_tcp_count+=1
            elif x[0] == udp:
                w_udp_count+=1
            elif x[0] == btcp:
                w_btcp_count+=1

            #print('wrong'+str(i)+', global:'+str(x[0])+'\n MSE'+str(x[1]))
            pass
    #print (count,data_num)
    print('accuracy:  '+str((total_count/data_num)*100)+'%')

    #print('accuracy:  '+str((total_count/data_num)*100)+'%   '+str(tcp_count)+'/'+str(g_tcp_count)+'  '+str(udp_count)+'/'+str(g_udp_count)+'   '+str(btcp_count)+'/'+str(g_btcp_count))
    #print('inaccuracy:'+str((w_total_count/data_num)*100)+'%   '+str(w_tcp_count)+'/'+str(g_tcp_count)+'   '+str(w_udp_count)+'/'+str(g_udp_count)+'   '+str(w_btcp_count)+'/'+str(g_btcp_count))

    #write output as a file
    resultf= open("../result/prediction.txt",'w')
    
    for x in predictResult:
       resultf.write(str(x)+'\n') 
    
    print("--- %s seconds ---" % (time.time() - start_time))

