"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy
import cPickle
from pyll import scope

import pymongo as pm
import numpy as np
import scipy.stats as sp_stats

import hyperopt
from hyperopt import base

from thoreano.slm import slm_from_config, FeatureExtractor
from dldata_classifier import train_scikits

import simffa_mtDat as mtdat
import simffa_fboDat as fbo
from yamutils.stats import pearsonr, spearmanr

from simffa_utils import slm_memmap 
from simffa_utils import slm_h5
from skdata import larray                                  
import simffa_params as sp
import simffa_analysisFns as sanal

## Pruned Bandits ##

@base.as_bandit()
def SimffaL1Bandit_pruned(template, label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL1GaborBandit(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL1GaborLargerBandit(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL2Bandit(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL2GaborBandit_pruned(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL3Bandit_pruned(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaL3GaborBandit_pruned(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

@base.as_bandit()
def SimffaV1LikeBandit_pruned(label_id=None, shuf=False):
    return scope.bandit_evaluatePsyFace_pruned(template, label_id, shuf)

####################
########Common stuff
####################

def get_features(X, config, verbose=False, dirname=None, fname_tag=None):
    features = slm_h5(
                    desc=config['desc'],
                    X=X,
                    basedir=dirname,
                    name=fname_tag, 
                    save=False) 
    return features
    
@scope.define
def bandit_evaluatePsyFace_pruned(config=None, label_id=None, shuf=False):

    dataset = mtdat.MTData_March082013()
    imgs,labels = dataset.get_images(label_id)
    nIm = labels.shape[0]
    print 'Loading ' + str(nIm) + 'imgs..'
    features = get_features(imgs, config, verbose=False)

    features = np.array(features)

    labels2 = dataset.get_labels(2)
    labels3 = dataset.get_labels(3)
    labels4 = dataset.get_labels(4)

    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    record = {}
    record['spec'] = config
    record['num_features'] = num_features
    record['feature_shape'] = fs
    
    num_splits = 4
    num_train = np.floor(0.75*nIm)
    num_test = np.floor(0.10*nIm)
    seed = np.random.randint(num_splits)
    
    psy_rsq_mu = regression_traintest(features, labels, seed, num_train, num_test, num_splits)
    psy_rsq_mu_shuf = regression_traintest(features, labels[shuf_ind], seed, num_train, num_test, num_splits)
    
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - psy_rsq_mu
    record['status'] = 'ok'
    return record

                            
