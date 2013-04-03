import os
import cPickle

import pymongo as pm
import numpy as np
import tabular as tb
import scipy.signal as signal
import scipy.stats as stats

from scipy.stats.stats import pearsonr
from scipy.stats.stats import nanmean

import simffa_analysisFns as sfns
from hyperopt.mongoexp import MongoJobs, as_mongo_str
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

def scatter_metric(Jobs, exps):
    
    metrics = ['cluster', 'blob', 'topog']

    for i in range(3):
        exp_key = exps[i]
        C = list(Jobs.find({'exp_key':exp_key,'state':2}))
        psyCorr_mu = np.array([c['result']['results']['psyRegress'] for c in C])
        # psyCorr_mu = np.array([c['result']['results']['psyCorr_mu'] for c in C])
        t0 = (psyCorr_mu >= -1) & (psyCorr_mu <= 1)

        if psyCorr_mu.shape[0] == 0:
            continue
        plt.figure()

        for j in range(3):
            ax1 = plt.subplot(1,3,j+1)    
            psy_metric = np.array([c['result']['results']['psyCorr_' + metrics[j]] for c in C])
            t = ~np.isnan(psy_metric) & t0
            r,p = pearsonr(psy_metric[t], psyCorr_mu[t])
            leg = metrics[j] + ', L'+str(i+1)+', r=' + '{:.3}'.format(r)

            if (p < 0.05):
                print leg
            
            plt.plot(psy_metric[t], psyCorr_mu[t], '.', label=leg)
            # plt.plot(psy_metric, psyCorr_mu, 'o')
            plt.xlabel(metrics[j])
            plt.title('r=' + '{:.3}'.format(r))
            x0,x1 = ax1.get_xlim()
            y0,y1 = ax1.get_ylim()
            ax1.set_aspect((x1-x0)/(y1-y0))
            plt.locator_params(axis = 'x', nbins = 5)
            plt.locator_params(axis = 'y', nbins = 5)
        # plt.legend()

def sort_maps(Jobs, exps):
    
    metrics = ['cluster', 'blob', 'topog']

    for i in range(len(exps)):
        exp_key = exps[i]
        C = list(Jobs.find({'exp_key':exp_key,'state':2}))
        nJobs = np.array(C).shape[0]
        if nJobs == 0:
            continue

        psyCorr = np.array([c['result']['results']['psyCorr'] for c in C])
        if psyCorr.shape[0] == 0:
            continue
        # psyCorr_mu = np.array([nanmean(np.array(psyCorr[i]).ravel())**2 for i in range(nJobs)])
        psyCorr_mu = np.array([c['result']['results']['psyRegress'] for c in C])
        
        print psyCorr.shape[0]
        s_i = [i[0] for i in sorted(enumerate(psyCorr_mu), key=lambda x:x[1])]

        count = 0
        print str(len(s_i)) + ' jobs'
        stepSize =  int(np.ceil(len(s_i)/10))
        plt.figure()
        for i in range(0,len(s_i), stepSize):
            s_oi = s_i[i]
            psyCorr_curr = psyCorr[s_oi]
            count = count+1
            if count > 2:
                break
            plt.subplot(2,1,count)
            plt.imshow(psyCorr_curr)
            tmp = sfns.topographicProcuct(psyCorr_curr)
            # plt.title('{:.4}'.format(psyCorr_mu[s_oi]))
            plt.title('{:.4}'.format(tmp))
            plt.colorbar()
