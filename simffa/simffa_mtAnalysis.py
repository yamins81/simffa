import os
import cPickle

import pymongo as pm
import numpy as np
import tabular as tb
import scipy.signal as signal
import scipy.stats as stats
from scipy.stats.stats import pearsonr

from hyperopt.mongoexp import MongoJobs, as_mongo_str
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_plots():
    conn = pm.Connection('localhost', 22334)
    db = conn['simffa']
    Jobs = db['jobs']

    exps = ['SimffaL1BanditRandom', 'SimffaL2BanditRandom', 'SimffaL3BanditRandom']
    metrics = ['cluster', 'blob', 'topog']
    
    for i in range(3):
        plt.figure()
        exp_key = exps[i]
        C = list(Jobs.find({'exp_key':exp_key,'state':2}))
        psyCorr_mu = np.array([c['result']['results']['psyCorr_mu'] for c in C])
        t0 = ~np.isnan(psyCorr_mu)
        for j in range(3):
            ax1 = plt.subplot(1,3,j+1)    
            psy_metric = np.array([c['result']['results']['psyCorr_' + metrics[j]] for c in C])
            t = ~np.isnan(psy_metric) & t0
            plt.plot(psy_metric[t], psyCorr_mu[t], 'o')
            r,p = pearsonr(psy_metric[t], psyCorr_mu[t])
            plt.title('r=' + '{:.4}'.format(r))
            plt.xlabel(metrics[j])
            
            x0,x1 = ax1.get_xlim()
            y0,y1 = ax1.get_ylim()
            ax1.set_aspect((x1-x0)/(y1-y0))
            # plt.locator_params(axis = 'x', nbins = 5)
            # plt.locator_params(axis = 'y', nbins = 5)
        plt.savefig('L' + str(i+1) + '_mu.png')

        # plt.figure()
        # psyCorr_mu = np.array([c['result']['results']['psyRegress'] for c in C])
        # for j in range(3):
        #     plt.subplot(1,3,j+1)    
        #     psy_metric = np.array([c['result']['results']['psyCorr_' + metrics[j]] for c in C])
        #     plt.plot(psy_metric, psyCorr_mu, 'o')
        #     r,p = pearsonr(psy_metric, psyCorr_mu)
        #     plt.title('r = ' + str(r))
        #     plt.xlabel(metrics[j])
        # plt.savefig('L' + str(i+1) + '_reg.png')

make_plots()
