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
import myplot as myplt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

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
    slope, intercept, r_value, p_value, std_err = linregress(fitC, yhat)
    act_Csurf   = fitC[0] # actual Csurf
    ndp         = fitdepth.shape[0] # number of data points in fitting
    d_org = 0 if Zsurf == depth[0] else Zsurf
    d_tot       = depth[-1]  # total thickness
    if plott == 1:
        return {'fitting':[yhat,fitC,fitdepth], 
                'prop':[profid,zstar,Csurf,act_Csurf,ndp,Cmin,Zmin,d_org,d_tot], 
                'stat':[z_r2, z_rmse, z_pcterr, p_value]}
    else:
        return {'prop':[profid,zstar,Csurf,act_Csurf,ndp,Cmin,Zmin,d_org,d_tot], 
                'stat':[z_r2, z_rmse, z_pcterr, p_value]}

        
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
    for profid, profname in enumerate(uniqprofname[:100]):
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
            if (not isinstance(res, list)):
                if plott != 0:
                    out_fitting.append(res['fitting'])
                    print 'plot profid: ', profid
                    fign = (profid+1)%ppf
                    if profid%ppf == 0:
                        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))
                    ax = fig.axes[fign-1]
                    ax.invert_yaxis()
                    ax.scatter(pctC, depth-depth[pctC<20][0], label='obs', c='g')
                    ax.plot(res['fitting'][0], res['fitting'][2], label='fitted')
                    ax.set_xlabel('pctC(%) ' + str(profid) + ':(' + profname + ')')
                    ax.set_ylabel('depth (cm)')
                    txtprop = {'fontsize':9,'horizontalalignment':'left',
                               'verticalalignment':'bottom','transform':ax.transAxes}
                    ax.text(.7, .2, 'z* = %.2f'%(res['prop'][1]),**txtprop)
                    ax.text(.7, .3, 'r2 = %.2f'%(res['stat'][0]),**txtprop)
                    ax.text(.7, .4, 'rmse = %.2f'%(res['stat'][1]),**txtprop)
                    ax.text(.7, .5, 'p-val = %.2f'%(res['stat'][3]),**txtprop)
                    plt.legend(loc=2,fontsize=9)
                    #pylab.text(.8, .8, df.loc[profname]['ecoregion'],fontsize=8)
                    if fign == 0:
                        print 'save plot, profid ', profid
                        plt.tight_layout()
                        if 'exp' in kwargs and kwargs['exp'] == 0:
                            fig.savefig(pathh + 'figures\\lmfit\\fig_%s.png'%(profid))   
                        else:
                            fig.savefig(pathh + 'figures\\expfit\\fig_%s.png'%(profid))   
                        plt.close()
                out_prop.append(res['prop'])
                out_stat.append(res['stat'])
            if isinstance(res, list):
                failed.append(res)
                if plott != 0:
                    # still save or creat new fig when current profile failed
                    if profid%ppf == 0:
                        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,8))
                    fign = (profid+1)%ppf
                    plt.legend(loc=2,fontsize=9)  
                    plt.tight_layout()
                    if fign == 0:
                        print 'save plot, profid ', profid
                        plt.tight_layout()
                        if 'exp' in kwargs and kwargs['exp'] == 0:
                            fig.savefig(pathh + 'figures\\lmfit\\fig_%s.png'%(profid))   
                        else:
                            fig.savefig(pathh + 'figures\\expfit\\fig_%s.png'%(profid))   
                        plt.close()                    
        else:
            failed.append([norawly,profid])
            if plott != 0:
                # still save or creat new fig when current profile failed
                if profid%ppf == 0:
                    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,8))
                fign = (profid+1)%ppf
                plt.legend(loc=2,fontsize=9)  
                plt.tight_layout()
                if fign == 0:
                    print 'save plot, profid ', profid
                    plt.tight_layout()
                    if 'exp' in kwargs and kwargs['exp'] == 0:
                        fig.savefig(pathh + 'figures\\lmfit\\fig_%s.png'%(profid))   
                    else:
                        fig.savefig(pathh + 'figures\\expfit\\fig_%s.png'%(profid))   
                    plt.close()
    return out_fitting, out_prop, out_stat, failed
   
#%%
if __name__ == "__main__":
    import pandas as pd
    #pathh       = 'C:\\download\\work\\!manuscripts\\C14_synthesis\\JenMeetingApr10_Zstar\\'
    pathh = 'C:\\Users\\happysk8er\\Google Drive\\manuscripts\\C14_synthesis\\JenMeetingApr10_Zstar\\'        
    layerfn     = 'ISCN datasets\\ISCNLayerData_LATEST.csv'
    proffn      = 'ISCN datasets\\ISCNProfileData_LATEST.csv'
    layerdata   = pd.read_csv(pathh+layerfn,encoding='iso-8859-1',index_col='profile_name')  
    profdata    = pd.read_csv(pathh+proffn,encoding='iso-8859-1',index_col='profile_name')  
    uniqprofname = layerdata.index.unique()

    # run exp fit    
    out_fitting_exp, out_prop_exp, \
        out_stat_exp, failed_exp = mainrun(layerdata, profdata, uniqprofname, plott=0)
    out_stat_exp = np.array(out_stat_exp)
    out_prop_exp = np.array(out_prop_exp)
    np.savez(pathh+'out_exp.npz',out_stat_exp=out_stat_exp, out_prop_exp=out_prop_exp, failed_exp=failed_exp) 
    dum = np.c_[out_stat_exp, out_prop_exp]
    df = pd.DataFrame(data=dum,
                      columns=['r2','rmse','pcterr','p_val',
                               'profid','zstar','Csurf_fit','Csurf_obs',
                               'N','Cmin','Zmin','d_org','d_tot'])
    tmp = np.array([i.encode('ascii','ignore') for i in uniqprofname])
    df.insert(0,'prof_name',tmp[out_prop_exp[:,0].astype(int)])
    df.to_csv(pathh+'ISCNzstar\\exp.csv',index=False)    
    
    # run linear fit
    out_fitting_lm, out_prop_lm, \
        out_stat_lm, failed_lm = mainrun(layerdata, profdata, uniqprofname, exp=0, plott=0)
    out_stat_lm = np.array(out_stat_lm)
    out_prop_lm = np.array(out_prop_lm)
    np.savez('out_lm',out_stat_lm=out_stat_lm, out_prop_lm=out_prop_lm, failed_lm=failed_lm) 


