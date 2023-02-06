
'''Sequentially Plot energy density samples based on toy energy functions using flows and mh algorithms'''

import numpy as np
import torch
import random
import torch.optim as optim
from torch.autograd import Variable
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'Flows_scripts'))
import nn as nn_
import flows
import matplotlib.pyplot as plt
import time
import seaborn as sns
import random
import argparse
from utils import set_all_seeds, kldiv
import pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'MH_Samplers'))
from MH_sampler_basic import mh
from toy_energy import U1, U2, U3, U4, U5, U6, U7, U8, U9
parser = argparse.ArgumentParser(description='inputs_toyenergy')
parser.add_argument('--out',default = '/mnt/home/premchan/Normalizing-Flows-Review/', type=str, help='path to results')
parser.add_argument('--method', default = 'True', type=str, help='which algorithm to use; "MH" - metropolis or "NAF" - Flows or "True" - true density')
args = parser.parse_args()


#Dictionary of parameters used for each model
#Left to right: Energy function, number of epochs for flows, nsamps for MH, hidden layer dimension, Type of NAF flow, approximated normalization constant, seed

toy_energy={"U1":[U1,15000,400000,5, "DSF",6.5371,2],
            "U2":[U2,15000,400000,5, "DSF",27.3805,3],
            "U3":[U3,15000,400000,2,"DSF", 31.5069,3],
            "U4":[U4,15000,400000,5,"DSF", 3.0456,1],
            "U5":[U5,15000,400000,5, "DSF",5.6198,0],
            "U6":[U6,15000,400000,5,"DSF",13.9084,1],
            "U7":[U7,15000,400000,8,"DSF",9.8347,2],
            "U8":[U8,15000,400000,8,"DSF",11.8317,2],
            "U9":[U9,15000,400000,16,"DSF",30.7431,0]
            }

class model(object):
    
    def __init__(self, target_energy, flow_type, ds_dim):
#        self.mdl = nn_.SequentialFlow( 
#                flows.IAF(2, 128, 1, 3), 
#                flows.FlipFlow(1), 
#                flows.IAF(2, 128, 1, 3),
#                flows.FlipFlow(1), 
#                flows.IAF(2, 128, 1, 3))

        self.ds_dim = ds_dim

        if flow_type == "DSF":
            self.mdl = flows.IAF_DSF(2, 64, 1, 4,
                num_ds_dim=self.ds_dim, num_ds_layers=2)
        else:
            self.mdl = flows.IAF_DDSF(2, 64, 1, 4,
                num_ds_dim=self.ds_dim, num_ds_layers=2)

        
        self.optim = optim.Adam(self.mdl.parameters(), lr=0.0005, 
                                betas=(0.9, 0.999))
        
        self.target_energy = target_energy
        
    def train(self,epochs):
        
        total = epochs
        loss_store=[]
        for it in range(total):

            self.optim.zero_grad()
            
            spl, logdet, _ = self.mdl.sample(64)
            losses = - self.target_energy(spl) - logdet
            loss = losses.mean()
            
            loss.backward()
            self.optim.step()

            loss_store.append(loss.detach().numpy())


            
            if ((it + 1) % 1000) == 0:
                print ('Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data))
        return loss_store
             


# build and train
fig = plt.figure(figsize=(10,10))
for i,key in enumerate(toy_energy): 
    print("Key",key)   
    ef = toy_energy[key][0]
    epochs = toy_energy[key][1]
    nsamps = toy_energy[key][2]
    ds_dim = toy_energy[key][3]
    flow_type = toy_energy[key][4]
    seed = toy_energy[key][6]
    print("seed",seed)
    set_all_seeds(seed)
    

           

    '''True Density'''  
    if args.method == "True":
        n=200       
        ax = fig.add_subplot(3,3,i+1)
        ax.set_title(key,fontsize=22,fontweight="bold")
        x = np.linspace(-10,10,n)
        y = np.linspace(-10,10,n)
        xx,yy = np.meshgrid(x,y)
        X = np.concatenate((xx.reshape(n**2,1),yy.reshape(n**2,1)),1)
        X = X.astype('float32')
        X1 = Variable(torch.from_numpy(X))
        Z = np.exp(ef(X1).data.numpy()).reshape(n,n)/toy_energy[key][5]
        ax.pcolormesh(xx,yy,np.exp(Z))
        ax.axis('off')
        plt.xlim((-10,10))
        plt.ylim((-10,10))

    '''MCMC'''
    if args.method == "MH":   
        mh_samples,accept = mh(nsamps, 2, ef, burn_in=0)
        idx = [i for i in range(0,mh_samples.shape[0],40)]
        mh_samples = mh_samples[idx,:].numpy()       
        #Visualization
        XX = mh_samples[:,0]
        YY = mh_samples[:,1]
        ax = fig.add_subplot(3,3,i+1)
        plot = ax.hist2d(XX,YY,200,range=np.array([(-10, 10), (-10, 10)]),density=True) 
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.axis('off')

    '''flows'''
    if args.method=="NAF":
        mdl = model(ef, flow_type, ds_dim)
        loss_store = mdl.train(epochs)
        data = mdl.mdl.sample(10000)[0].data.numpy()
        #Plot flows 
        ax = fig.add_subplot(3,3,i+1)
        XX = data[:,0]
        YY = data[:,1]
        plot = ax.hist2d(XX,YY,200,range=np.array([(-10, 10), (-10, 10)]),density=True)
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.axis('off')
        
plt.savefig(args.out+'model_toy_all'+args.method+'.png')
plt.clf()



