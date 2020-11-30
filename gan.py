import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from helpers import *
from classes import *
import math
import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

             
torch.cuda.set_device(3)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dur=[]
epochs = 10000
train_num = 20
test_num =2
lrg = 0.0005
lrd = 0.001
DIS_TRAIN_NUM = 1
GEN_TRAIN_NUM = 5
enc = drums_encoder_decoder.MultiDrumOneHotEncoding()

name = "lstmgan"
sessions = ["session1","session2","session3"]
gen = lstmd().to(device)
dis = lstmDis().to(device)
g_optimizer = torch.optim.Adam(gen.parameters(),lr = lrg)
d_optimizer = torch.optim.Adam(dis.parameters(),lr = lrd)
writerb = SummaryWriter('runs/{}'.format(name))

for epoch in range(epochs):
    t0 = time.time()
    gen.train()
    dis.train()
    dis_acc_list = []
    dis_loss_list = []
    
    for i in range(DIS_TRAIN_NUM):
        ## pick midi
        current = os.getcwd()
        path = os.path.join(current, "data") 
        ##batch
        batch_size = 200 #####################hi
        in_tensor = []
        ground_truth = []
        for j in range(batch_size):
            tpath = os.path.join(path ,"drummer" +str(random.randint(1,9))) 
            ## choose session
            tpath = os.path.join(tpath, random.choice(os.listdir(tpath)))
            ## choose midisa
            midi_path = os.path.join(tpath, random.choice(os.listdir(tpath))) 
            ##input array making
            try:
                in_tensor.append(midi_to_input(midi_path).reshape(-1,8*9))
            except notLong:
                continue
        
        in_tensor = torch.tensor(in_tensor).to(device=device, non_blocking=True).float()
#         ground_truth = torch.tensor(ground_truth).to(device=device, non_blocking=True).float().unsqueeze(0)

        thickness = in_tensor.shape[0]
        noise = torch.randn(1,thickness,256).to(device)
        fake_samples = gen(noise)
        dis_input = torch.cat([in_tensor,fake_samples],axis=0)
        labels = torch.cat([torch.ones(thickness,1),torch.zeros(thickness,1)],axis=0).to(device) ## 0 = fake
        
        #forward discrimnator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        output = dis(dis_input)
        train_loss = F.binary_cross_entropy(output,labels) 
        #backward
        train_loss.backward()
        d_optimizer.step()
        guesses = torch.round(output)
        correctness = torch.eq(guesses,labels).double()
        dis_acc_list.append(torch.mean(correctness).item())
        dis_loss_list.append(train_loss.item())
        
    gen_acc_list = []
    gen_loss_list = []
    for i in range(GEN_TRAIN_NUM):
        ##batch
        batch_size = 200 #####################hi
        in_tensor = []
        noise = torch.randn(1,batch_size,256).to(device)
        fake_samples = gen(noise)
        labels = torch.ones(batch_size,1).to(device) ## 0 = fake
        
        #forward genrator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        output = dis(fake_samples)
        train_loss = F.binary_cross_entropy(output,labels) 
        #backward
        train_loss.backward()
        g_optimizer.step()
        guesses = torch.round(output)
        correctness = torch.eq(guesses,labels).double()
        gen_acc_list.append(torch.mean(correctness).item())
        gen_loss_list.append(train_loss.item())

    dis_loss = np.mean(dis_loss_list)
    dis_acc = np.mean(dis_acc_list)
    gen_loss = np.mean(gen_loss_list)
    gen_acc = np.mean(gen_acc_list)
        
    ## eval
    if epoch % 600 ==0:
        # noise = torch.randn(1,32)
        with torch.no_grad():
            noise = torch.normal(torch.zeros(1,1,256), std=1).to(device)
            music = np.rint(gen.forward(noise,num=8).cpu().detach().numpy()).astype(int)[0]
            music = np.expand_dims(music.flatten(),0)

            drums = drums_lib.DrumTrack()
            for i in range(1152//9):
                target = music[0,i*9:(i+1)*9]
                res = 0
                for ele in target: 
                    res = (res << 1) | ele
            #     print(res)
                drums.append(enc.decode_event(res))
            lol=drums.to_sequence()
            midi_io.sequence_proto_to_midi_file(lol,'./ganmusic/{}_{}.midi'.format(name,str(epoch)))
        
    
    ##lol
    print("Epoch {:05d} | dis_loss {:.4f} | gen_loss {:.4f}  |dis_acc {:.4f} | gen_acc {:.4f}  | Time(s) {:.4f}" .format(epoch, dis_loss, gen_loss,dis_acc,gen_acc, np.mean(dur)))
    writerb.add_scalar('dis_loss',dis_loss,epoch)
    writerb.add_scalar('dis_acc',dis_acc,epoch)
    writerb.add_scalar('gen_loss',gen_loss,epoch)
    writerb.add_scalar('gen_acc',gen_acc,epoch)
  
             
    dur.append(time.time() - t0)
    torch.save(gen.state_dict(), "models/{}_gen.pth".format(name))
    torch.save(dis.state_dict(), "models/{}_dis.pth".format(name))