import time
import operator
start_time = time.time()

data_dir = '11-20_no_under_max_25'
gtf = open("../data/"+data_dir+"/tweetEmoji.csv")#ground truth
outf = open("../data/"+data_dir+"/usedIndiceStat.txt","w")#ground truth
outf2 = open("../data/"+data_dir+"/usedIndice.txt","w")#ground truth

wordUsedIndice={}
for i,l in enumerate(gtf):
    line=l.rstrip().split(',')

    if i%10000 == 0:
        print i
        print("--- %s seconds ---" % (time.time() - start_time))

    for x in [int(x) for x in line[1].split()]:
        if x in wordUsedIndice:
            wordUsedIndice[x] = wordUsedIndice[x]+1
        else:
            wordUsedIndice[x] = 1




sortedIndice = sorted(wordUsedIndice.items(), key=operator.itemgetter(1), reverse = True)


#print wordUsedIndice
print len(wordUsedIndice)
for x in sortedIndice:
    outf.write(str(x)+'\n')
    outf2.write(str(x[0])+'\n')
