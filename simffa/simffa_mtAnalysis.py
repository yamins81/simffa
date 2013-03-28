import os
import cPickle

import pymongo as pm
import numpy as np
import tabular as tb
import scipy.signal as signal
import scipy.stats as stats

from scipy.stats.stats import pearsonr
from scipy.stats.stats import nanmean

from hyperopt.mongoexp import MongoJobs, as_mongo_str
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

def scatter_metric(Jobs, exps):
    
    plt.figure()
    metrics = ['cluster', 'blob', 'topog']

    for i in range(3):
        exp_key = exps[i]
        C = list(Jobs.find({'exp_key':exp_key,'state':2}))
        # psyCorr_mu = np.array([c['result']['results']['psyRegress'] for c in C])
        psyCorr_mu = np.array([c['result']['results']['psyCorr_mu'] for c in C])
        t0 = (psyCorr_mu >= -1) & (psyCorr_mu <= 1)

        for j in range(3):
            ax1 = plt.subplot(1,3,j+1)    
            psy_metric = np.array([c['result']['results']['psyCorr_' + metrics[j]] for c in C])
            t = ~np.isnan(psy_metric) & t0
            r,p = pearsonr(psy_metric[t], psyCorr_mu[t])
            leg = 'L'+str(i+1)+', r=' + '{:.3}'.format(r)
            
            plt.plot(psy_metric[t], psyCorr_mu[t], 'o', label=leg)
            # plt.plot(psy_metric, psyCorr_mu, 'o')
            plt.xlabel(metrics[j])
            
            x0,x1 = ax1.get_xlim()
            y0,y1 = ax1.get_ylim()
            ax1.set_aspect((x1-x0)/(y1-y0))
            plt.locator_params(axis = 'x', nbins = 5)
            plt.locator_params(axis = 'y', nbins = 5)
        # plt.legend()

def sort_maps(Jobs, exps):
    
    metrics = ['cluster', 'blob', 'topog']

    for i in range(3):
        exp_key = exps[i]
        C = list(Jobs.find({'exp_key':exp_key,'state':2}))
        nJobs = np.array(C).shape[0]
        if nJobs == 0:
            continue

        psyCorr = np.array([c['result']['results']['psyCorr'] for c in C])
        psyCorr_mu = np.array([nanmean(np.array(psyCorr[i]).ravel()) for i in range(nJobs)])
        # psyCorr_mu = np.array([c['result']['results']['psyRegress'] for c in C])
        
        s_i = [i[0] for i in sorted(enumerate(psyCorr_mu), key=lambda x:x[1])]

        count = 0
        stepSize =  int(np.ceil(len(s_i)/10))
        plt.figure()
        for i in range(0,len(s_i), stepSize):
            s_oi = s_i[i]
            psyCorr_curr = psyCorr[s_oi]
            count = count+1
            if count > 10:
                break
            plt.subplot(2,5,count)
            plt.imshow(psyCorr_curr)
            plt.title('{:.4}'.format(psyCorr_mu[s_oi]))
            plt.colorbar()

# conn = pm.Connection('localhost', 22334)
# db = conn['simffa']
# Jobs = db['jobs']
# exps = ['SimffaL1BanditRandom', 'SimffaL2BanditRandom', 'SimffaL3BanditRandom']
# scatter_metric(Jobs, exps)
# sort_maps(Jobs, exps)
# plt.show()
