# -*- coding: utf-8 -*-
"""
script file of comparing exp fit and lm fit on log transformed data

Created on Wed Jun 24 10:44:19 2015

@author: Yujie
"""
import numpy as np
import matplotlib.pyplot as plt
import myplot as myplt
import pandas as pd

expnpzfiles = np.load('out_exp.npz')
lmnpzfiles  = np.load('out_lm.npz')
#%% statistics
#% plot hist of exp and lm
expr2   = expnpzfiles['out_stat_exp'][:,0]
exprmse = expnpzfiles['out_stat_exp'][:,1]
exppcte = expnpzfiles['out_stat_exp'][:,2]
lmr2   = lmnpzfiles['out_stat_lm'][:,0]
lmrmse = lmnpzfiles['out_stat_lm'][:,1]
lmpcte = lmnpzfiles['out_stat_lm'][:,2]

# plot historgram of all profiles that are fitted using exp vs. lm
expr2cl = filter(lambda x:~np.isinf(x) and ~np.isnan(x) and x<1. and x>0., expr2)
lmr2cl = filter(lambda x:~np.isinf(x) and ~np.isnan(x) and x<1. and x>0., lmr2)
exprmsecl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), exprmse)
lmrmsecl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), lmrmse)
exppctecl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), exppcte)
lmpctecl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), lmpcte)

#  plot r2
plt.hist(lmr2cl,bins=20,normed=1,alpha=0.5, cumulative=True,label='lm fit (N = ' + str(len(lmr2cl)) + ')')
plt.hist(expr2cl,bins=20,normed=1,alpha=0.5, cumulative=True,label='exp fit (N = ' + str(len(expr2cl)) + ')')
plt.gca().set_xlabel('r2')
plt.gca().set_ylabel('cumulative Fraction (%)')
plt.legend(loc=0)   

# plot rmse

plt.hist(lmrmsecl,bins=20,normed=1,alpha=0.5, cumulative=True,label='lm fit (N = ' + str(len(lmrmsecl)) + ')')
plt.hist(exprmsecl,bins=20,normed=1,alpha=0.5, cumulative=True,label='exp fit (N = ' + str(len(exprmsecl)) + ')')
plt.gca().set_xlabel('rmse')
plt.gca().set_ylabel('cumulative Fraction (%)')
plt.legend(loc=0)   

# plot pcte
plt.hist(lmpctecl,bins=20,normed=1,alpha=0.5, cumulative=True,label='lm fit (N = ' + str(len(lmpctecl)) + ')')
plt.hist(exppctecl,bins=20,normed=1,alpha=0.5, cumulative=True,label='exp fit (N = ' + str(len(exppctecl)) + ')')
plt.gca().set_xlabel('pcte')
plt.gca().set_ylabel('cumulative Fraction (%)')
plt.legend(loc=0)   


#%% properties. z* etc.
# plot z* using exp vs. lm
expprofid  = expnpzfiles['out_prop_exp'][:,0]
expzstar   = expnpzfiles['out_prop_exp'][:,1]
expcsurf   = expnpzfiles['out_prop_exp'][:,2]
lmprofid    = lmnpzfiles['out_prop_lm'][:,0]
lmzstar    = lmnpzfiles['out_prop_lm'][:,1]
lmcsurf    = lmnpzfiles['out_prop_lm'][:,2]

#  that are fitted using exp vs. lm
expzstarcl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), expzstar)
lmzstarcl = filter(lambda x:~np.isinf(x) and ~np.isnan(x), lmzstar)
lm_df = pd.DataFrame(data=expnpzfiles['out_prop_exp'][:,1:],
                     index=expnpzfiles['out_prop_exp'][:,0],columns=['zstar','csurf'])
exp_df = pd.DataFrame(data=lmnpzfiles['out_prop_lm'][:,1:],
                     index=lmnpzfiles['out_prop_lm'][:,0],columns=['zstar','csurf'])
join_df = exp_df.join(lm_df, how='inner', lsuffix='_exp', rsuffix='_lm')

fig, axes = plt.subplots(figsize=(10,8))
axes.scatter(join_df['zstar_exp'], join_df['zstar_lm'])
axes.set_ylim([-500,1000])
axes.set_xlim([-500,1000])
axes.set_xlabel('zstar (cm, exp)')
axes.set_ylabel('zstar (cm, lm)')
myplt.refline()

fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(join_df['csurf_exp'], join_df['csurf_lm'])
axes.set_ylim([0,60])
axes.set_xlim([0,60])
axes.set_xlabel('csurf (%, exp)')
axes.set_ylabel('csurf (%, lm)')
myplt.refline()
