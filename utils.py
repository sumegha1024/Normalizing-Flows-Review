import numpy as np
import torch
import random


c = - 0.5 * np.log(2*np.pi)
def log_normal(x, mean, log_var, eps=0.00001):
    return - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var/2. + c    

def varify(x):
    return torch.autograd.Variable(torch.from_numpy(x))

def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max    
    return B

def set_all_seeds(seed):
  random.seed(seed)
  #os.environ('PYTHONHASHSEED') = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Function to Calculate Sample KL-divergence
def kldiv(X, Y, reg_X=1e-10, reg_Y=1e-10):
    Xnorm = X / X.sum()
    Xreg = X + reg_X
    Xreg /= Xreg.sum()
    Yreg = Y + reg_Y
    Yreg /= Yreg.sum()
    s1 = (Xnorm * np.log(Xreg / Yreg)).sum()
    return s1


def pr_or_roc_curve(ground_truth, scores,pr=True):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = ground_truth[sorted_indices]

    thresholds = np.unique(sorted_scores)
    n_thresholds = len(thresholds)

    precision = np.zeros(n_thresholds)
    recall = np.zeros(n_thresholds)
    tp = np.zeros(n_thresholds)
    fp = np.zeros(n_thresholds)
    fn = np.zeros(n_thresholds)
    tn = np.zeros(n_thresholds)

    for i, threshold in enumerate(thresholds):
        tp[i] = np.sum((sorted_scores >= threshold) & (sorted_labels == 1))
        fp[i] = np.sum((sorted_scores >= threshold) & (sorted_labels == 0))
        fn[i] = np.sum((sorted_scores < threshold) & (sorted_labels == 1))
        tn[i] = np.sum((sorted_scores < threshold) & (sorted_labels == 0))
        precision[i] = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] != 0 else 1.0
        recall[i] = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] != 0 else 1.0

    if pr==True:
        return precision, recall, thresholds
    else: 
        return fp/(fp+tn), recall, thresholds

def area_under_curve(x, y):
    area = np.trapz(x, y)
    return area

