#data_path = 'animation'
data_path = 'real'
#data_path = '1211_tcp_udp'
#data_path = '1211_tcp_bad'


f = open("../data/"+data_path+'/bps_result.txt')

maxLen = 0
data =[] 
data_num = 0
maxLenWords = []
for l in f:
    data_num+=1
    line=l.rstrip().split(',')

    if len(line)-1 > maxLen:
        maxLen = len(line)-1
        data = line


print('max_len',maxLen)
print('data:',data)
print('data_num:',data_num)
