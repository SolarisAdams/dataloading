import matplotlib.pyplot as plt
from numpy import mean
import os

root = "/home/Adama/dataloading/"
reset = 1
files = os.listdir(root)

for filename in files:
    if filename.endswith(".txt"):
        tt = os.stat(root + filename).st_mtime
        model = filename.split(".")[0]
        if os.path.exists(root + "speed-"+model+".png"):
            tp = os.stat(root + "speed-"+model+".png").st_mtime
            if tp > tt and reset==0:
                continue
        if "-128" in filename:
            batch_size = 128
        elif "-256" in filename:
            batch_size = 256
        elif "-512" in filename:
            batch_size = 512
        elif "vgg13" in filename or "resnet50" in filename or "-384" in filename:
            batch_size = 384
        else:
            batch_size = 1024
        # batch_size=128

        def seq(l, t):
            a = []
            b = 1.0
            
            for i in range(l):
                a.append(b)
                b *= t
            a = a / (mean(a) * l)
            return a

            


        def avg(a, index, l=1):
            if l > index+1:
                sequence = seq(index+1, 0.97)
            else:
                sequence = seq(l, 0.97)
            s = 0
            p = index
            for t in sequence:
                s += t*a[p]
                p -= 1

            return s

        def map(a):
            
            b = []
            for i in range(len(a)):
                b.append(batch_size/avg(a, i))
            return b


        y_data = []
        with open("/home/Adama/dataloading/"+model+".txt", "r") as f:
            for line in f:
                time, speed = line.split()
                y_data.append(float(time))

        if len(y_data)>300:
            y_data = y_data[0:300]

        # y_data2 = []
        # with open("/home/Adama/dataloading/"+model+"-us.txt", "r") as f:
        #     for line in f:
        #         time, speed = line.split()
        #         y_data2.append(float(speed))
        # y_data2 = y_data2[0:len(y_data)]

        x_data = list(range(len(y_data)))

        avgy_data = map(y_data)
        # print(y_data)
        # plt.plot(x_data,y_data,color='red',linewidth=0.5)
        # plt.plot(x_data,y_data2,color='blue',linewidth=0.5)
        plt.plot(x_data,avgy_data,color='red',linewidth=0.5)
        plt.show()
        plt.savefig(root + "speed-"+model+".png")
        plt.close()
