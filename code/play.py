import time


inf = open('play_data.csv','r')
for j,l in enumerate(inf):
    l = l.rstrip().split(',')

    d=[[],[],[],[],[],[]]
    for i in range(6):
        d[i].append(float(1-float(l[2+6*i])))
        d[i].append(float(1-float(l[3+6*i])))
        d[i].append(float(1-float(l[4+6*i])))


    print('------'+str(j/10.0)+'secs-------')
    time.sleep(0.1)
    print('Flow    Norm  UDP   Bad TCP      Classified')   

      
    for i in range(6):

        label = 'Normal TCP'

        p = max( d[i][0],d[i][1],d[i][2])

        if p >0.975:
            if p == d[i][0]:
                label = 'Normal TCP'
            elif p == d[i][2]:
                label = 'Bad TCP'
            elif p == d[i][1]:
                label = 'UDP'


        

        print('Flow'+str(i+1)+'\t'+"%.3f %.3f %.3f"%(d[i][0],d[i][1],d[i][2])+'\t'+label)



