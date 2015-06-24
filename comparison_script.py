# -*- coding: utf-8 -*-
"""
script file of comparing exp fit and lm fit on log transformed data

Created on Wed Jun 24 10:44:19 2015

@author: Yujie
"""
import numpy as np
import matplotlib.pyplot as plt

expnpzfiles = np.load('out_exp.npz')
lmnpzfiles  = np.load('out_lm.npz')
#%% plot hist of exp and lm
expr2   = expnpzfiles['out_stat_exp'][:,0]
exprmse = expnpzfiles['out_stat_exp'][:,1]
exppcte = expnpzfiles['out_stat_exp'][:,2]
lmr2   = lmnpzfiles['out_stat_lm'][:,0]
lmrmse = lmnpzfiles['out_stat_lm'][:,1]
lmpcte = lmnpzfiles['out_stat_lm'][:,2]

expr2cl = filter(lambda x:~np.isinf(x) and ~np.isnan(x) and x<1. and x>0., expr2)
lmr2cl = filter(lambda x:~np.isinf(x) and ~np.isnan(x) and x<1. and x>0., lmr2)

plt.hist(lmr2cl,bins=20,normed=1,alpha=0.5, cumulative=True,label='lm fit (N = ' + str(len(lmr2cl)) + ')')
plt.hist(expr2cl,bins=20,normed=1,alpha=0.5, cumulative=True,label='exp fit (N = ' + str(len(expr2cl)) + ')')
plt.gca().set_xlabel('r2')
plt.gca().set_ylabel('cumulative Fraction (%)')
plt.legend(loc=0)   
