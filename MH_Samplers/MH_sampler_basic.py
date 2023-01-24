'''Implementing the MH algorithm'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
import random
#import statsmodels.api as sm

'''Return accept with probability p and reject with probability 1-p'''
def random_coin(p):
    p = float(p)
    unif = random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True

'''Sampling algorithm'''    
def mh(hops,data_dim, pot, burn_in, sigma=2, log_trans=True):
    states = torch.zeros(hops,data_dim)
    
    #Intitialize current state from the proposal distribution
    
    current = torch.Tensor(1,data_dim).normal_(0,sigma)
    accept =[]
    for i in range(hops):
        states[i,:] = current
        
        #Sample from proposal distribution
         
        movement = torch.Tensor(1,data_dim).normal_(0,sigma) + current

        #Evaluate acceptance ratio
        if log_trans:
            curr_prob = torch.exp(pot(current))
            move_prob = torch.exp(pot(movement))
        else:
            curr_prob = pot(current)
            move_prob = pot(movement)

        acceptance = min(move_prob/curr_prob,1) 
        
        if random_coin(acceptance):
            current = movement
            accept.append(1)
        else:
            accept.append(0)
    return states[burn_in:hops,:], sum(accept)/len(accept)

