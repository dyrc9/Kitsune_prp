from Kitsune import Kitsune
import numpy as np
import time

# File location
path = "./pcaps/mirai.pcap.tsv" #the pcap, pcapng, or tsv file to process.


# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)
packet_limit = FMgrace + ADgrace #the number of packets to process, here we only need to get model, so we don't need to execute

# Build Kitsun
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune. building a model:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    if i % 1000 == 0:
        print(i)
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse) 
    #we don't need to save RMSEs, but to make the codes more like the oringin version, we still save the rmse
stop = time.time()
print("Building model completes. Time elapsed: "+ str(stop - start))

K.savemodel()




