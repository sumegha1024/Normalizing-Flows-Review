
'''In this we get a numerical integral for toy energy density functions'''

import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy import integrate
from scipy.special import log_softmax


delta = 1e-6

c = - 0.5 * np.log(2*np.pi)
def log_normal(x, mean, log_var, eps=0.00001):
    return - (x-mean) ** 2 / (2. * np.exp(log_var) + eps) - log_var/2. + c    


def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - log_softmax(inputs,axis=1)).mean(dim, keepdims=keepdim)
    

def sigmoid(Z):
    return 1/(1+(np.exp(-Z)))

def U1(z1,z2):
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])

    E1 = 0.5 * (((z1**2+z2**2)**0.5 - 2.) / 0.4) ** 2
    E2 = - np.log(np.exp(-0.5 * ((z1-2)/0.6)**2) + 
                 np.exp(-0.5 * ((z1+2)/0.6)**2))   
    return np.exp(- (E1 + E2))
 
def U2(z1,z2):
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])

    E1 = 0.5 * (((0.5*(z1**2+z2**2))**0.5 - 2.) / 0.5) ** 2
    E2 = - np.log(np.exp(-0.5 * ((z1-2)/0.6)**2) + 
                 np.exp(-0.5 * ((np.sin(z1)/0.5))**2) + 
                 np.exp(-0.5 * ((z1+z2+2.5)/0.6)**2))   
    return np.exp(- (E1 + E2))
  
def U3(z1,z2):
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])

    R = 2.0
    return np.exp(- 1.*(R-(z1**2+.5*z2**2)**0.5)**2)
    
 ###EDIT FROM HERE   

def U4(z1, z2, small=False):

    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])
    
    z1 = z1*2
    z2 = z2*2
    Z = np.vstack((z1,z2)).transpose()

    if small:
        mean = np.array([[-2., 0.],
                                [2., 0.],
                                [0., 2.],
                                [0., -2.]],
                                dtype='float32')
        lv = np.log(np.ones(1)*0.2)
    else:
        mean = np.array([[-5., 0.],
                                [5., 0.],
                                [0., 5.],
                                [0., -5.]],
                                dtype='float32')
        lv = np.log(np.ones(1)*1.5)
    
    d1 = log_normal(Z, mean[None,0,:], lv).sum(1)+np.log(0.1)
    d2 = log_normal(Z[:,:], mean[None,1,:], lv).sum(1)+np.log(0.3)
    d3 = log_normal(Z[:,:], mean[None,2,:], lv).sum(1)+np.log(0.4)
    d4 = log_normal(Z[:,:], mean[None,3,:], lv).sum(1)+np.log(0.2)

    
    return np.exp(logsumexp(np.concatenate(
        [d1[:,None],d2[:,None],d3[:,None],d4[:,None]],1),1) + 2.5)
    

def U5(z1,z2):
    
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])

    w1 = np.sin(z1*0.5*math.pi)
    return np.exp(- 0.5*((z2-w1)/0.4)**2  - 0.1 * (z1**2))


def U6(z1,z2):

    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])


    w1 = np.sin(z1*0.5*math.pi)
    w2 = 3*np.exp(-0.5*(z1-2)**2)

    return np.exp(logsumexp(np.concatenate(
            [(-0.5*((z2-w1)/0.35)**2)[:,None],
             (-0.5*((z2-w1+w2)/0.35)**2)[:,None]], 1), 1) - 0.05 * (z1**2))



def U7(z1,z2):

    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])

    w1 = np.cos(z1*0.35*math.pi)
    w2 = 2*np.exp(-0.2*((z1-2)*2)**2)
    return np.exp(logsumexp(np.concatenate(
            [(-0.5*((z2-w1)/0.35)**2)[:,None],
             (-0.5*((z2-w1+w2)/0.35)**2)[:,None]], 1), 1) - 0.1 * ((z1-2)**2)) 


def U8(z1,z2):
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])
    
    Z = np.vstack((z1,z2)).transpose()
    w1 = np.sin(z1*0.5*math.pi)
    w3 = 2.5*sigmoid((z1-2)/0.3)

    return np.exp(logsumexp(np.concatenate(
          [(logsumexp(np.concatenate(
                  [(-0.5*((z2-w1)/0.4)**2)[:,None],
                   (-0.5*((z2-w1+w3)/0.35)**2)[:,None]], 1), 1) - 
                    0.05 * ((z1-2)**2+(z2+3)**2))[:,None] ,
         -2.5*(Z[:,0:1]-2)**2 - 2.5*(Z[:,1:2]-2)**2], 1), 1))


def U9(z1,z2):
    if not (isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray)):
        z1 = np.array([z1])
        z2 = np.array([z2])
    
    Z = np.vstack((z1,z2)).transpose()
    w1 = np.sin(z1*0.5*math.pi)
    w3 = 2.5*sigmoid((z1-2)/0.3)
    return np.exp(logsumexp(np.concatenate(
        [(logsumexp(np.concatenate(
                  [(-0.5*((z2-w1)/0.4)**2)[:,None],
                   (-0.5*((z2-w1+w3)/0.35)**2)[:,None]], 1), 1) - 
                    0.05 * ((z1)**2+(z2)**2))[:,None] ,
         2*U3(z1*1.5-2,z2*1.5-2)[:,None]], 1), 1) -1)
   


if __name__ == '__main__': 
    



    #print(integrate.nquad(U9, [[-np.inf, np.inf],[-np.inf, np.inf]]))
    
    #Visualizing the energy function
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    integral_store = []
    mm, MM = -5, 5
    
    n = 300
    # plot figure
    fig = plt.figure(figsize=(10,10))
    
    for j in range(1,9+1):
        integral = eval('integrate.nquad(U{}, [[-np.inf, np.inf],[-np.inf, np.inf]])'.format(j))
        print("INTEGRAL",integral)
        integral_store.append(integral[0])
        ax = fig.add_subplot(3,3,j)
        x = np.linspace(mm,MM,n)
        y = np.linspace(mm,MM,n)
        xx,yy = np.meshgrid(x,y)
        X = np.concatenate((xx.reshape(n**2,1),yy.reshape(n**2,1)),1)
        X = X.astype('float32')
        Z = eval('U{}(X[:,0],X[:,1])'.format(j)).reshape(n,n)/integral[0]
        #plt.pcolormesh(xx,yy,np.exp(Z), cmap='RdBu_r')#), norm=colors.NoNorm())
        sns.heatmap(np.exp(Z)[::-1], ax=ax, cmap="YlGnBu", cbar=False) #YlGnBu
        plt.axis('off')
        #plt.xlim((mm,MM))
        #plt.ylim((mm,MM))
    
    plt.tight_layout()
    plt.savefig('targets_check.png')
    print("INTEGRALSS",integral_store)





