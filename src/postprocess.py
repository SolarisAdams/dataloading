import os

root = "/home/Adama/dataloading/loading/"
files = os.listdir(root)
sleep = 0
start = 40

with open(root + "avg.ans","w") as fo:
    for filename in files:
        if filename.endswith(".txt") and not "full-time" in filename:
            if "-128" in filename:
                batch_size = 128
            elif "-256" in filename:
                batch_size = 256
            elif "-512" in filename:
                batch_size = 512
            elif "-768" in filename:
                batch_size = 768
            elif "-2048" in filename:
                batch_size = 2048
            elif "-1536" in filename:
                batch_size = 1536
            elif "-1024" in filename:
                batch_size = 1024
            elif "vgg13" in filename or "resnet50" in filename or "-384" in filename:
                batch_size = 384
            elif "alexnet" in filename:
                batch_size = 1024
            else:
                batch_size = 512


            # batch_size = 512


            with open (root + filename, "r") as fi:
                ans = 0
                p = 0
                total_time = 0
                for line in fi:
                    time, speed = line.split()[0:2]
                    p += 1
                    if p > start:
                        total_time += float(time)
                if p > start:
                    ans = (p-start)*batch_size/total_time
            print(filename[:-4] + "\t", ans, file=fo)
            print(filename[:-4], "\t", ans, "\t", batch_size)
            



