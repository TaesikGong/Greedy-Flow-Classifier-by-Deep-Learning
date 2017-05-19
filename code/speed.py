import time

st= time.time()
i=0
for x in range(10000000):
    pass 
print time.time()-st
for x in range(100000000):
    pass 
print time.time()-st
for x in range(1000000000):
    pass

print time.time()-st

for x in range(1000000000):
    pass 
print time.time()-st

