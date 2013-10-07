
import numpy as np
import scipy.stats as sp_stats
from yamutils.stats import pearsonr, spearmanr
from dldata_classifier import train_scikits

from pyll import scope
import hyperopt
from hyperopt import base
import simffa_params as sp
from simffa_utils import get_features
from simffa_utils import save_features

import simffa.simffa_dataset_fbo as fbo
# import dldata.HvM.neural_datasets as hvm

def getFSI(features, labels):
    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])
    mu_f = (features[labels,:]).mean(0)
    mu_nf = (features[~labels,:]).mean(0)
    
    FSI = (mu_f - mu_nf) / (np.abs(mu_f + mu_nf)) 
    return FSI

# @scope.define
# def fbo_bandit_evaluateFSI_HvM(config=None):
    
#     dataset = fbo.FaceBodyObject20110803()
#     imgs, labels = dataset.get_images()
#     fbo_features = sb.get_features(imgs, config)
#     FSI = fbo.getFSI(fbo_features, labels)
    
#     dataset = hvm.HvMWithDiscfade()
#     imgs = dataset.get_images(, {'size': (400, 400), 'global_normalize': True})
#     hvm_features = sb.get_features(imgs, config)
    
#     attachments = {}
#     attachments['hvm_features'] = hvm_features
#     attachments['fbo_features'] = fbo_features
#     attachments['FSI'] = FSI
    
#     fn = save_features('/hyperopt_features/facegen_fsi/', attachments)

#     results = {}
#     results['feature_fn'] = fn
#     results['FSI'] = FSI
    
#     record = {}
#     record['spec'] = config
#     record['results'] = results
#     record['attachments'] = {}
#     record['loss'] = 0
#     record['status'] = 'ok'
#     return record

@scope.define
def fbo_bandit_evaluateFSI(config=None):
    
    dataset = fbo.FaceBodyObject20110803()
    imgs, labels = dataset.get_images()
    fbo_features = get_features(imgs, config)
    FSI = getFSI(fbo_features, labels)
    
    # attachments = {}
    # attachments['fbo_features'] = fbo_features
    # attachments['FSI'] = FSI
    
    # fn = save_features('/hyperopt_features/facegen_fsi/', attachments)

    results = {}
    # results['feature_fn'] = fn
    results['FSI'] = FSI
    
    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 0
    record['status'] = 'ok'
    return record


@base.as_bandit()
def Simffa_FboFSI_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.fbo_bandit_evaluateFSI(template)

@base.as_bandit()
def Simffa_FboFSI_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.fbo_bandit_evaluateFSI(template)

@base.as_bandit()
def Simffa_FboFSI_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.fbo_bandit_evaluateFSI(template)

# general case
@base.as_bandit()
def Simffa_FboFSI_Bandit(template=None):
    return scope.fbo_bandit_evaluateFSI(template)
                  
