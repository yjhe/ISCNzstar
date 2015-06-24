# -*- coding: utf-8 -*-
"""
Script for Z*. Processing ISCN dataset
direct exponential fit to pctC

Created on Fri Jun 19 10:43:46 2015

@author: happysk8er
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from numba import autojit
import mystats as mysm

#@autojit
def expfunc(x, K, I):
    '''
    Z* function. not forcing through Csurf. 
    @params: K, I
        z*      = -1/K
        csurf   = I
    '''    
    return I*np.exp(K*x)

def linfunc(x, a, b):
    '''
    Z* function. not forcing throught Csurf. pass in log trans pctC
    z*      = -1/b
    csurf   = np.exp(a)
    '''    
    return a + b*x

#@autojit  
def zstarfunc(depth, pctC, profid, profname, plott=0, exp=1):
    '''
    Pass in observations of each profile, fit func, return yhat (if plott=1), zstar, stat
    parameters:
        exp      : by default, fit exponential func directly. otherwise, fit 
                   linear  funct on log transformed pctC
    output: 
        fitted value: 1-d vec 
        failure code: 
            -1   : no mineral soil
            -2   : layer number < 3, inside zstarfunc
            -3   : no avalable layer, in raw data
            -999 : optimization failed
    '''
    from scipy.optimize import curve_fit
    print 'fit profile ', profid
    
    # define failure code
    nomin    = -1
    toofewly = -2
    optifd   = -999       
    
    if min(pctC) >= 20: # no minearl soil layer
        return [nomin, profid]        
    Csurf         = pctC[pctC<20][0] # defined by not used. override with fitted value
    Zsurf         = depth[pctC<20][0]
    depth_mineral = depth - Zsurf # depth vec starts from Zsurf
    Cmin          = np.nanmin(pctC)
    Zmin          = np.nanmax(depth_mineral[pctC==Cmin]) # max or min?    
    idx           = np.logical_and(depth_mineral >= 0, depth_mineral <= Zmin)
    idx           = np.logical_and(idx, pctC>0)
    fitdepth      = depth_mineral[idx]
    fitC          = pctC[idx]
    nlayer        = fitdepth.shape[0]    
    if nlayer < 3:
        return [toofewly, profid]

    try:
        if exp == 1:
            popt, pcov = curve_fit(expfunc, fitdepth, fitC, maxfev=500, 
                                   p0=(-0.01,fitC[0]))        
        else:
            popt, pcov = curve_fit(linfunc, fitdepth, np.log(fitC), maxfev=500)        
    except RuntimeError:
        print 'optimization failed for profid ', profid
        return [optifd, profid]

    if exp == 1:
        Csurf       = popt[1]
        zstar       = -1./popt[0]
        yhat        = expfunc(fitdepth, *popt)
    else:
        Csurf       = np.exp(popt[0])
        zstar       = -1./popt[1]
        yhat        = np.exp(linfunc(fitdepth, *popt))

    # plt.plot(fitdepth, fitC);plt.plot(yhat, fitdepth)
    z_r2        = mysm.cal_R2(fitC, yhat)
    z_rmse      = mysm.cal_RMSE(fitC, yhat)
    z_pcterr    = mysm.cal_pctERR(fitC, yhat)
    if plott == 1:
        return {'fitting':[yhat,fitC,fitdepth], 'prop':[profid,zstar,Csurf], 
                'stat':[z_r2, z_rmse, z_pcterr]}
    else:
        return {'prop':[profid,zstar,Csurf], 'stat':[z_r2, z_rmse, z_pcterr]}
        
def mainrun(layerdata, profdata, uniqprofname, plott=0, ppf=9, **kwargs):
    '''
    fit Z* function to ISCN dataset
    parameters:
        pathh: path to the folder under which ISCN datasets and figures are at.
        plott: whether to plot (1) or not (default, 0)
        ppf: plots per figure, default 9
        failure code: 
            -1   : no mineral soil
            -2   : layer number < 3, inside zstarfunc
            -3   : no avalable layer, in raw data
            -999 : optimization failed
    '''
    norawly     = -3
    out_fitting = []
    out_prop    = []
    out_stat    = []
    failed      = []
    for profid, profname in enumerate(uniqprofname):
        rawdepth    = np.array(layerdata.loc[profname]['layer_bot_cm'])
        rawpctCtot  = np.array(layerdata.loc[profname]['c_tot_percent'])
        rawpctCoc   = np.array(layerdata.loc[profname]['oc_percent'])
        if rawpctCtot[~np.isnan(rawpctCtot)].shape[0] > rawpctCoc[~np.isnan(rawpctCoc)].shape[0]:
            rawpctC = rawpctCtot
        else:
            rawpctC = rawpctCoc
        notNaNs = ~np.isnan(rawdepth) & ~np.isnan(rawpctC) 
        depth = rawdepth[notNaNs]; pctC = rawpctC[notNaNs]
        if depth.shape[0] > 0:
            res = zstarfunc(depth, pctC, profid, profname, plott=plott, **kwargs)
            if (not isinstance(res, float)) and (not isinstance(res, list)):
                if plott != 0:
                    out_fitting.append(res['fitting'])
                    print 'plot profid: ', profid
                    fign = (profid+1)%ppf
                    if profid%ppf == 0:
                        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))
                    ax = fig.axes[fign-1]
                    ax.invert_yaxis()
                    ax.scatter(pctC, depth, label='obs', c='g')
                    ax.plot(res['fitting'][:,0], res['fitting'][:,2], label='fitted')
                    ax.set_xlabel('pctC(%) ' + str(profid) + ':(' + profname + ')')
                    ax.set_ylabel('depth (cm)')
                    plt.legend()
                    #pylab.text(.8, .8, df.loc[profname]['ecoregion'],fontsize=8)
                    if fign == 0:
                        print 'save plot, profid ', profid
                        plt.tight_layout()
                        fig.savefig(pathh + 'figures\\fig_%s.png'%(profid))    
                        plt.close()
                out_prop.append(res['prop'])
                out_stat.append(res['stat'])
            if isinstance(res, list):
                failed.append(res)
        else:
            failed.append([norawly,profid])
    return out_fitting, out_prop, out_stat, failed
   
#%%
if __name__ == "__main__":
    pathh       = 'C:\\download\\work\\!manuscripts\\C14_synthesis\\JenMeetingApr10_Zstar\\'
    #pathh = 'C:\\Users\\happysk8er\\Google Drive\\manuscripts\\C14_synthesis\\JenMeetingApr10_Zstar\\'        
    layerfn     = 'ISCN datasets\\ISCNLayerData_LATEST.csv'
    proffn      = 'ISCN datasets\\ISCNProfileData_LATEST.csv'
    layerdata   = pd.read_csv(pathh+layerfn,encoding='iso-8859-1',index_col='profile_name')  
    profdata    = pd.read_csv(pathh+proffn,encoding='iso-8859-1',index_col='profile_name')  
    uniqprofname = layerdata.index.unique()

    # run exp fit    
    out_fitting_exp, out_prop_exp, \
        out_stat_exp, failed_exp = mainrun(layerdata, profdata, uniqprofname)
    out_stat_exp = np.array(out_stat_exp)
    out_prop_exp = np.array(out_prop_exp)
    np.savez('out_exp',out_stat_exp=out_stat_exp, out_prop_exp=out_prop_exp, failed_exp=failed_exp) 
    
    # run linear fit
    out_fitting_lm, out_prop_lm, \
        out_stat_lm, failed_lm = mainrun(layerdata, profdata, uniqprofname, exp=0)
    out_stat_lm = np.array(out_stat_lm)
    out_prop_lm = np.array(out_prop_lm)
    np.savez('out_lm',out_stat_lm=out_stat_lm, out_prop_lm=out_prop_lm, failed_lm=failed_lm) 
    
            