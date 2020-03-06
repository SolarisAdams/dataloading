n = 1000
r = 0.003
data = []

with open("loading/disk.ans","r")as f:
    for line in f:
        data.append(float(line))
print(max(data))
t = [0 for _ in range(n)]
for num in data:
    index = int(num*n/r)
    if index > n-1:
        index = n-1
    t[index] += 1
with open("loading/disk-fb.ans","w")as f:
    for i,tt in enumerate(t): 
        print(i*r/n, "\t", tt,file=f, sep="")
        
