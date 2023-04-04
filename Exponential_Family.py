'''Exponential Family Examples'''
import numpy as np
import torch
import random
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.join(sys.path[0],'Flows_scripts'))
import nn as nn_
import flows
import matplotlib.pyplot as plt
import time
import seaborn as sns
import random
import argparse
import scipy.stats as ss
from utils import set_all_seeds
import math
import statsmodels.api as sm
import pandas as pd
from scipy.interpolate import splrep, splev
import matplotlib.lines as mlines
#from utils import kldiv


parser = argparse.ArgumentParser(description='inputs_exp')
parser.add_argument('--lr', default = 0.0005, type=float, help='learning rate')
parser.add_argument('--seed',default =2, type=int, help='seed for simulation')
parser.add_argument('--out',default = '/mnt/home/premchan/Normalizing-Flows-Review/Temp2/', type=str, help='path to results')
parser.add_argument('--n_data',default =50, type=int, help='number of samples for observed y')
parser.add_argument('--sigma',default =1.0, type=float)
args = parser.parse_args()

set_all_seeds(0)


'''Simulate dataset for normal, inverse-gamma conjugate family'''
global n,nt,tau, example, alpha, beta
example=1
nw=args.n_data
mu=np.random.uniform(0,1)
print("mu",mu)
sigma = args.sigma
yw=np.random.normal(0,1,nw)*sigma+mu
print("Yw",yw)
idx=set([item for item in range(nw)])
S=random.sample(idx,int(nw*0.8))
St=list(idx.difference(S))
y=torch.tensor(yw[S])
yt=torch.tensor(yw[St])
n=y.shape[0]
nt=yt.shape[0]
tau=1.0 #y ~N(mu,sigma**2) and mu ~ N(0,tau**2) and sigma**2 ~ ig(2.5/2,2.5/2)
set_all_seeds(args.seed)
print("SAMPLE SIZE",n)
time_store_normal={"MCMC":None,"Flows":None,"VI":None}

'''Gibbs Sampling'''
start = time.time()
nsweep = 11000
sigma_samples_MCMC_store = np.zeros(nsweep)
mu_samples_MCMC_store = np.zeros(nsweep)
mu_MCMC = np.mean(y.numpy())
for i in range(nsweep):
    #Update steps
    ytilda = y.numpy() - mu_MCMC
    sigma_MCMCsq = ss.invgamma.rvs(a=(2.5/2+n/2),scale=(2.5/2+np.sum(ytilda**2)/2))
    Sig_MCMC=1.0/(n/sigma_MCMCsq+1/tau**2)
    mu_MCMC=np.random.normal(0,1)*np.sqrt(Sig_MCMC) + Sig_MCMC*np.sum(y.numpy())/sigma_MCMCsq 
    #store values
    mu_samples_MCMC_store[i] = mu_MCMC
    sigma_samples_MCMC_store[i] = sigma_MCMCsq


time_store_normal["MCMC"]=time.time() - start

mu_samples_MCMC_store = mu_samples_MCMC_store[1000:]
sigma_samples_MCMC_store = sigma_samples_MCMC_store[1000:]

#Diagnostic Plots

fig = plt.figure()       
ax = fig.add_subplot(1,2,1)
ax.plot(mu_samples_MCMC_store)
ax = fig.add_subplot(1,2,2)
ax.plot(sigma_samples_MCMC_store)
plt.savefig(args.out+'trace_seed'+str(args.seed)+'.png')
plt.clf()

fig, ax = plt.subplots(1,2)  
sm.graphics.tsa.plot_acf(pd.DataFrame(mu_samples_MCMC_store), lags=10, ax = ax[0])
sm.graphics.tsa.plot_acf(pd.DataFrame(sigma_samples_MCMC_store), lags=10, ax = ax[1])
plt.savefig(args.out+"Autocorr_normal_ig.png")
plt.clf()


'''Mean Field VI for Normal-Normal,Ig example'''
def elbo(params): #returns negative ELBO
    m1=params[0]
    rho1=params[1]
    s1 = torch.log(1 + torch.exp(rho1))
    sigma_v1 = params[2]
    sigma_v2 = params[3]
    v1 = torch.log(1 + torch.exp(sigma_v1))
    v2 = torch.log(1 + torch.exp(sigma_v2))
    nll = 0.5*n*np.log(2*math.pi) + 0.5*n*(torch.log(v2)-torch.digamma(v1))
    nll = nll + 0.5*(v1/v2)*(torch.sum(y**2) - 2*torch.sum(y)*m1 + n*(s1**2 + m1**2)) #negative log likelihood 
    ap = torch.Tensor([2.5/2]) #Hyper parameters for prior of sigma**2 ~ IG(2.5/2,2.5/2)
    bp = torch.Tensor([2.5/2])
    kl_ig = (v1 - ap)*torch.digamma(v1) - torch.lgamma(v1) + torch.lgamma(ap) + ap*(torch.log(v2) - torch.log(bp)) + v1*((bp-v2)/v2)
    kl_gauss = 0.5*(np.log(tau**2) - torch.log(s1**2)) -0.5 +0.5*(m1**2 + s1**2)/tau**2
    loss= nll + kl_ig + kl_gauss
    return loss

params=torch.zeros(4)
params[0] = torch.mean(y)
params[1] = torch.Tensor(1).uniform_(-1.0,-1.0)
params[2:4] = torch.Tensor(2).uniform_(1.0,1.0)
#print(params)

params.requires_grad_()
n_optim_steps = int(10000)
optimizer = torch.optim.Adam([params], 5e-2)
loss_store=[]
start = time.time()
for ii in range(n_optim_steps):
    optimizer.zero_grad()
    loss = elbo(params)
    loss_store.append(loss.detach().numpy())
    if ii%1000 ==0:
        print('Step # {}, loss: {}'.format(ii, loss.item()))
    loss.backward()
    # Access gradient if necessary
    optimizer.step()
time_store_normal["VI"]=time.time() - start

#Generate samples for VI
m1_vi, rho1_vi = torch.Tensor(params[0].detach()), torch.Tensor(params[1].detach())
sigma_v1_vi, sigma_v2_vi = torch.Tensor(params[2].detach()), torch.Tensor(params[3].detach())
s1_vi = torch.log(1 + torch.exp(rho1_vi))
v1_vi = torch.log(1 + torch.exp(sigma_v1_vi))
v2_vi = torch.log(1 + torch.exp(sigma_v2_vi))

mu_samples_vi = (s1_vi*torch.randn(10000) + m1_vi).numpy()
sigma_samples_vi = ss.invgamma.rvs(v1_vi.numpy(), size=10000)*(v2_vi.numpy())

#Loss plot

plt.plot(loss_store)
plt.savefig(args.out+'loss1vi_seed'+str(args.seed)+'.png')
plt.clf()
'''Some useful functions to set up Flows'''

def gaussian_log_pdf(z):
      #  - z: a batch of m data points (size: m x no. of params) tensor

    return -.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(1)

def invgamma_log_pdf(z,a,b):
       # - z: a batch of m data points (size: m) tensor   
    a = torch.Tensor([a])
    b = torch.Tensor([b])
    return a*torch.log(b) - torch.lgamma(a) -(a+1)*torch.log(z) - b/z

def beta_log_pdf(z,a,b):
     # - z: a batch of m data points (size: m) tensor
    a = torch.Tensor([a])
    b = torch.Tensor([b])
    return (a-1)*torch.log(z) + (b-1)*torch.log(1-z) -torch.lgamma(a)-torch.lgamma(b) + torch.lgamma(a+b)

def U5(z,s=None): #s is sigma_squared
    if s is None:
        return 0.5*(n*(z**2).squeeze()-2*torch.sum(y)*z.squeeze() + torch.sum(y**2))
    else:
        return 0.5*n*torch.log(s) + 0.5*(n*(z**2)-2*torch.sum(y)*z + torch.sum(y**2))/s

def U6(z,s = None): 
    if s is None:
        return 0.5*(z**2).squeeze()/tau**2
    else:
        return 0.5*(z**2)/tau**2 - invgamma_log_pdf(s,2.5/2,2.5/2)

exact_log_density_normal = lambda z,s=None: -U5(z,s)-U6(z,s)

class model(object):
    
    def __init__(self, target_energy, flow_type, ds_dim,p):
#        self.mdl = nn_.SequentialFlow( 
#                flows.IAF(2, 128, 1, 3), 
#                flows.FlipFlow(1), 
#                flows.IAF(2, 128, 1, 3),
#                flows.FlipFlow(1), 
#                flows.IAF(2, 128, 1, 3))

        self.ds_dim = ds_dim
        self.p=p

        if flow_type == "DSF":
            self.mdl = flows.IAF_DSF(self.p, 64, 1, 4,
                num_ds_dim=self.ds_dim, num_ds_layers=2)
        else:
            self.mdl = flows.IAF_DDSF(self.p, 64, 1, 4,
                num_ds_dim=self.ds_dim, num_ds_layers=2)

        
        self.optim = optim.Adam(self.mdl.parameters(), lr=args.lr, 
                                betas=(0.9, 0.999))
        
        self.target_energy = target_energy
        
    def train(self,epochs):
        
        total = epochs
        loss_store=[]
        time_axis=[]
        for it in range(total):

            self.optim.zero_grad()
            
            spl, logdet, _ = self.mdl.sample(32)
            if example ==1:
                losses = - self.target_energy(spl[:,0],(torch.log(1+torch.exp(spl[:,1])))**2) - logdet - torch.log(torch.exp(spl[:,1])/(1+torch.exp(spl[:,1]))) - torch.log(2*torch.log(1+torch.exp(spl[:,1]))) #Normal-normal
                #losses = - self.target_energy(spl[:,0],(torch.log(1+torch.exp(spl[:,1])))**2) - logdet - 2*torch.log(1+torch.exp(spl[:,1]))*torch.exp(spl[:,1])/(1+torch.exp(spl[:,1])) #Normal-normal
            else:
                losses = - self.target_energy(torch.exp(spl)/(1+torch.exp(spl))) - logdet - torch.log((torch.exp(spl)/((1+torch.exp(spl))**2)).squeeze()) #Bernoulli-Beta

            loss = losses.mean()
            
            loss.backward()
            self.optim.step()

            loss_store.append(loss.detach().numpy())


            
            if ((it + 1) % 1000) == 0:
                print ('Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data))
        return loss_store


'''Flows for Normal-Normal,Ig example'''

mdl = model(exact_log_density_normal, "DSF", 8,2)
start = time.time()
loss_store = mdl.train(15000)
time_store_normal["Flows"]=time.time() - start
data = mdl.mdl.sample(10000)[0].data.numpy()

'''Generate Plots'''
sns.kdeplot(mu_samples_MCMC_store,color='red',label='MCMC')
sns.kdeplot(mu_samples_vi,color="blue",label="MF-VI")
sns.kdeplot(data[:,0],color='green',label='Flows')
plt.legend()
plt.savefig(args.out+'mu'+'trial'+str(args.seed)+'.png')
plt.clf()

sns.kdeplot(sigma_samples_MCMC_store,color='red',label='MCMC')
sns.kdeplot(sigma_samples_vi,color="blue",label="MF-VI")
sns.kdeplot((np.log(1+np.exp(data[:,1])))**2,color='green',label='Flows')
plt.legend()
plt.savefig(args.out+'sigma'+'trial'+str(args.seed)+'.png')
plt.clf()

fig, ax = plt.subplots()
handles =[]
sns.kdeplot(data[:,0],(np.log(1+np.exp(data[:,1])))**2,color='green',label='Flows',ax=ax)
handles.append(mlines.Line2D([], [], color='green', label="Flows"))
sns.kdeplot(mu_samples_vi,sigma_samples_vi,color='blue',label='MF-VI',ax=ax)
handles.append(mlines.Line2D([], [], color='blue', label="MF-VI"))
sns.kdeplot(mu_samples_MCMC_store,sigma_samples_MCMC_store,color='red',label='Gibbs',ax=ax)
handles.append(mlines.Line2D([], [], color='red', label="Gibbs"))
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
ax.legend(handles = handles,fontsize=15)
plt.xlabel(r'$\mu$',fontsize=20)
plt.ylabel(r'$\sigma^{2}$',fontsize=20)
plt.savefig(args.out+'contour.pdf')
plt.clf()


plt.plot(loss_store)
plt.savefig(args.out+'loss1flows_seed'+str(args.seed)+'.png')
plt.clf()

loss_store_df=pd.DataFrame(loss_store)
loss_store_df.to_csv(args.out+"loss_flows_example1.csv")

'''Simulate dataset for Bernoulli, Beta conjugate family'''
#y ~ Bernoulli(p) and p ~ Beta(alpha,beta)
set_all_seeds(0)
nw=args.n_data
p=np.random.uniform(0,1)
print("p",p)
yw=np.random.binomial(1,p ,size=nw)
print("Yw",yw)
idx=set([item for item in range(nw)])
S=random.sample(idx,int(nw*0.8))
St=list(idx.difference(S))
y=torch.tensor(yw[S])
yt=torch.tensor(yw[St])
n=y.shape[0]
nt=yt.shape[0]
alpha=1
beta=1
set_all_seeds(args.seed)


time_store_bernoulli={"MCMC":"NA","Flows":None,"VI":None}
#start = time.time()
#p_samples_MCMC_store=np.random.beta(alpha+np.sum(y.numpy()),beta+n-np.sum(y.numpy()),size=10000)
#time_store_bernoulli["MCMC"]=time.time()-start

example = 2


'''Mean Field VI for Bernoulli-beta example'''
def elbo(params): #returns negative ELBO
    rho_ap=params[0]
    rho_bp=params[1]
    ap = torch.log(1 + torch.exp(rho_ap))
    bp = torch.log(1 + torch.exp(rho_bp))
    alpha1=torch.Tensor([alpha])
    beta1=torch.Tensor([beta])
    nll = torch.sum(y)*(torch.digamma(bp)-torch.digamma(ap)) + n*(torch.digamma(ap+bp)) -n*torch.digamma(bp)
    kl_beta = torch.lgamma(alpha1) + torch.lgamma(beta1) - torch.lgamma(alpha1+beta1) -torch.lgamma(ap) - torch.lgamma(bp) + torch.lgamma(ap+bp)
    kl_beta += (ap-alpha1)*torch.digamma(ap) + (bp-beta1)*torch.digamma(bp) +(alpha1-ap+beta1-bp)*torch.digamma(ap+bp)
    loss= nll + kl_beta
    return loss

params=torch.zeros(2)
params[0] = torch.Tensor(1).uniform_(1.0,1.0)
params[1] = torch.Tensor(1).uniform_(1.0,1.0)
#print(params)

params.requires_grad_()
n_optim_steps = int(10000)
optimizer = torch.optim.Adam([params], 5e-2)
loss_store=[]
start = time.time()
for ii in range(n_optim_steps):
    optimizer.zero_grad()
    loss = elbo(params)
    loss_store.append(loss.detach().numpy())
    if ii%1000 ==0:
        print('Step # {}, loss: {}'.format(ii, loss.item()))
    loss.backward()
    # Access gradient if necessary
    optimizer.step()
    

time_store_bernoulli["VI"]=time.time() - start

rho_ap_vi, rho_bp_vi = torch.Tensor(params[0].detach()), torch.Tensor(params[1].detach())
ap_vi = torch.log(1 + torch.exp(rho_ap_vi))
bp_vi = torch.log(1 + torch.exp(rho_bp_vi))
print("ap",ap_vi)
print("bp",bp_vi)
p_samples_vi = ss.beta.rvs(a=ap_vi.numpy(),b=bp_vi.numpy(), size=10000)

plt.plot(loss_store)
plt.savefig(args.out+'loss2vi_seed'+str(args.seed)+'.png')
plt.clf()


def exact_log_density_bernoulli(z):
    # - z: a batch of m data points (size: mx1) tensor
    z=z.squeeze()
    ll=torch.sum(y)*torch.log(z) + (n-torch.sum(y))*torch.log(1-z)
    return ll +beta_log_pdf(z,alpha,beta)

mdl = model(exact_log_density_bernoulli, "DSF", 8,1)
start = time.time()
loss_store = mdl.train(20000)
time_store_bernoulli["Flows"]=time.time() - start
data = mdl.mdl.sample(10000)[0].data.numpy()

'''Generate Plots''' 
xgrid = np.arange (0, 1, 0.01)
y = ss.beta.pdf(xgrid,alpha+np.sum(y.numpy()),beta+n-np.sum(y.numpy()))
plt.plot(xgrid,y,color="red",label='True Posterior')
#sns.kdeplot(p_samples_MCMC_store,color='red',label='MC Samples')
#sns.kdeplot(p_samples_vi,color="blue",label="MF-VI")
sns.kdeplot(np.exp(data[:,0])/(1+np.exp(data[:,0])),color='green',label='Flows')
plt.ylabel("Density",fontsize=20)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.legend(fontsize=13)
plt.savefig(args.out+'p_'+'trial'+str(args.seed)+'.pdf')
plt.clf()


#iters = [i for i in range(100)]
#elbo_store=-np.array(loss_store[0:100])
#plt.figure()
#bspl = splrep(iters,elbo_store,s=4)
#bspl_y = splev(iters,bspl)
#plt.plot(iters,bspl_y)
plt.plot(loss_store)
plt.ylabel("ELBO",fontsize=15)
plt.xlabel("Epochs",fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(args.out+'loss2flows_seed'+str(args.seed)+'.pdf')
plt.clf()

loss_store_df=pd.DataFrame(loss_store)
loss_store_df.to_csv(args.out+"loss_flows_example2.csv")



print("Seed",args.seed)
print("Time in s for normal-normal+ig",time_store_normal)
print("Time in s for bernoulli-beta",time_store_bernoulli)


