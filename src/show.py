import matplotlib.pyplot as plt
from numpy import mean

model = "resnet"




y_data = []
time_data = []
tottime = 0
with open("/home/Adama/dataloading/"+model+".txt", "r") as f:
    for line in f:
        time, speed = line.split()
        y_data.append(float(speed))
        time_data.append(tottime)
        tottime += float(time)

# y_data = y_data[0:80]

# y_data2 = []
# with open("/home/Adama/dataloading/"+model+"-us.txt", "r") as f:
#     for line in f:
#         time, speed = line.split()
#         y_data2.append(float(speed))
# y_data2 = y_data2[0:len(y_data)]

x_data = list(range(len(y_data)))


# print(y_data)
plt.plot(time_data,y_data,color='red',linewidth=0.5)
# plt.plot(x_data,y_data2,color='blue',linewidth=0.5)
# plt.plot(x_data,y_data,color='red',linewidth=0.5)
plt.show()
plt.savefig("speed-"+model+".png")