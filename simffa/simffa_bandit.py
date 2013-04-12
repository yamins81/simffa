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

## Bandits ##

@base.as_bandit()
def SimffaL1Bandit(label_id=None, shuf=False):
    template = sp.l1_params
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL1GaborBandit(label_id=None, shuf=False):
    template = sp.l1_params_gabor
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL1GaborLargerBandit(label_id=None, shuf=False):
    template = sp.l1_params_gabor_larger
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL2Bandit(label_id=None, shuf=False):
    template = sp.l2_params
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL2GaborBandit(label_id=None, shuf=False):
    template = sp.l2_params_gabor
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL3Bandit(label_id=None, shuf=False):
    template = sp.l3_params
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaL3GaborBandit(label_id=None, shuf=False):
    template = sp.l3_params_gabor
    return scope.bandit_evaluate(template, label_id, shuf)

@base.as_bandit()
def SimffaV1LikeBandit(label_id=None, shuf=False):
    template = sp.v1like_params    
    return scope.bandit_evaluate(template, label_id, shuf)

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
    features = np.array(features)
    return features
    
def get_regression_splits(labels, seed, ntrain, ntest, num_splits):
        nIm = np.array(labels).shape[0]
        rng = np.random.RandomState(seed)
        splits = {}
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []
            
            perm = rng.permutation(nIm)
            for ind in perm[:ntrain]:
                splits['train_' + str(split_id)].append(ind)
            for ind in perm[ntrain: ntrain + ntest]:
                splits['test_' + str(split_id)].append(ind)

        return splits

def regression_traintest(features, labels, seed=0, ntrain=80, ntest=30, num_splits=5, algo='pls.PLSRegression'):
    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])
    splits = get_regression_splits(labels, seed, ntrain, ntest, num_splits)
    results = []
    rsq = 0;
    for ind in range(num_splits):
        train_inds = np.array(splits['train_' + str(ind)])
        test_inds = np.array(splits['test_' + str(ind)])
        train_X = features[train_inds]
        test_X = features[test_inds]
        train_y = labels[train_inds]
        test_y = labels[test_inds]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, algo, regression=True)
        # result = train_scikits(train_Xy, test_Xy, 'linear_model.RidgeCV', regression=True)
        rsq = rsq+result[1]['test_rsquared']
    rsq = rsq / num_splits

    return rsq

def get_regression_results(features, labels):
    num_splits = 4
    nnan = ~np.isnan(labels)
    labels = labels[nnan]
    features = features[nnan,:]
    nIm = labels.shape[0]
    num_train = np.floor(0.75*nIm)
    num_test = np.floor(0.25*nIm)
    seed = np.random.randint(num_splits)
    
    psy_rsq_mu = regression_traintest(features, labels, seed, num_train, num_test, num_splits)

    # shuf_ind = np.random.random_integers(0,len(labels)-1,len(labels))
    # psy_rsq_mu_shuf = regression_traintest(features, labels[shuf_ind], seed, num_train, num_test, num_splits)

    return psy_rsq_mu

    
def regression_trainingError(features, labels):
    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])
    XY = (features, labels)
    result = train_scikits(XY, XY, 'linear_model.RidgeCV', regression=True)
    rsq = result[1]['test_rsquared']
    return rsq

@scope.define
def bandit_evaluate(config=None, label_id=None, shuf=False):
    dataset = mtdat.MTData_March082013()
    imgs,labels = dataset.get_images(label_id)
    nIm = labels.shape[0]
    print 'Loading ' + str(nIm) + 'imgs...'
    features = get_features(imgs, config, verbose=False)

    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    
    print 'regressing model on labels'
    regress_r2 = {}
    labelnames = ['face', 'eye']
    neuralname = ['pl', 'ml', 'al']

    for label_i in range(len(labelnames)):
        label = dataset.get_labels(label_i+1)
        regress_r2[labelnames[label_i]] = get_regression_results(features, label)
        print 'done ' + labelnames[label_i]

    for neural_i in range(len(neuralname)):
        curr_neural = dataset.get_neural_labels(neural_i+1)
        tmp = [get_regression_results(features, curr_neural[:,ei]) for ei in range(4)]
        regress_r2[neuralname[neural_i]] = tmp

    results = {}
    results['regress_r2'] = regress_r2

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - regress_r2['face']
    record['status'] = 'ok'
    return record

@scope.define
def bandit_evaluatePsyFace(config=None, label_id=None, shuf=False):

    dataset = mtdat.MTData_March082013()
    imgs,labels = dataset.get_images(label_id)
    nIm = labels.shape[0]
    print 'Loading ' + str(nIm) + 'imgs..'
    features = get_features(imgs, config, verbose=False)


    if shuf==True:
        t = np.random.random_integers(0,len(labels)-1,len(labels))
        labels = labels[t]

    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    record = {}
    record['spec'] = config
    record['num_features'] = num_features
    record['feature_shape'] = fs

    print 'computing regression error'
    num_splits = 4
    num_train = np.floor(0.75*nIm)
    num_test = np.floor(0.10*nIm)
    seed = np.random.randint(num_splits)
    shuf_ind = np.random.random_integers(0,len(labels)-1,len(labels))
    psy_rsq_mu = regression_traintest(features, labels, seed, num_train, num_test, num_splits)
    psy_rsq_mu_shuf = regression_traintest(features, labels[shuf_ind], seed, num_train, num_test, num_splits)
    
    # features = features.mean(3)
    # print 'computed map'
    # psyCorr = sanal.getPearsonCorr2D(features, labels)
    # nonnanCorr = np.array(psyCorr).ravel()
    # nonnanCorr = nonnanCorr[~np.isnan(nonnanCorr)]
    # hist_n, hist_x = np.histogram(nonnanCorr, bins=50)

    print 'storing stats'
    results = {}
    results['psyRegress'] = psy_rsq_mu
    results['psyRegress_shuf'] = psy_rsq_mu_shuf

    # results['psyCorr'] = psyCorr
    # results['psyCorr_hist_n'] = hist_n.tolist()
    # results['psyCorr_hist_x'] = hist_x[1:].tolist()
    # results['psyCorr_mu'] = np.abs(psyCorr).ravel().mean(0)
    # results['psyCorr_cluster'] = sanal.getClusterSize(psyCorr)
    # results['psyCorr_blob'] = sanal.getBlobiness(psyCorr)
    # results['psyCorr_topog'] = sanal.getTopographicOrg(psyCorr)

    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - psy_rsq_mu
    record['status'] = 'ok'
    return record

# def evaluate_just_FSI(dataset, config, train=False, **training_data):
#     record, FSI = evaluate_FSI(dataset, config, train=train, **training_data)
#     for k in ['F_s_avg', 'BO_s_avg', 'FSI_s_avg']:
#         spatial_averages[k] = record.pop(k)       
#     return record, FSI

# @scope.define
# def evaluate_FSI(config=None, features=None, labels=None, train=True, **training_data):
#     if features is None:
#         dataset = fbo.FaceBodyObject20110803()
#         imgs, labels = dataset.img_classification_task()
#         features = get_features(imgs, config, verbose=True)
        
#     features = features[:]
#     features = np.array(features)
#     fs = features.shape
#     num_features = fs[1]*fs[2]*fs[3]    
#     meta = dataset.meta
#     face_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] == 'Face']
#     bo_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] != 'Face']
    
#     record = {}
#     record['num_features'] = num_features
#     record['feature_shape'] = fs
#     thresholds = np.arange(0,1,.01)
#     record['thresholds'] = thresholds.tolist()
    
#     F = features[face_inds].mean(0)
#     BO = features[bo_inds].mean(0)
#     record['F_s_avg'] = F.mean(2).tolist()
#     record['BO_s_avg'] = BO.mean(2).tolist()
  
#     FSI = (F - BO) / (np.abs(F) + np.abs(BO))
#     FSI_counts = [len((FSI > thres).nonzero()[0]) for thres in thresholds]
#     FSI_fractions = [c/ float(num_features) for c in FSI_counts]
#     record['fsi_fractions'] = FSI_fractions
#     record['FSI_s_avg'] = FSI.mean(2).tolist()
#     record['Face_selective_s_avg'] = (FSI > .333).astype(np.float).mean(2).tolist()

#     F_rect = np.abs(features[face_inds]).mean(0)
#     BO_rect = np.abs(features[bo_inds]).mean(0)
#     FSI_rect = (F_rect - BO_rect) / (F_rect + BO_rect)
#     FSI_rect_counts = [len((FSI_rect > thres).nonzero()[0]) for thres in thresholds]
#     FSI_rect_fractions = [c/ float(num_features) for c in FSI_rect_counts]
#     record['rectified_fsi_fractions'] = FSI_rect_fractions
#     record['rectified_FSI_s_avg'] = FSI_rect.mean(2).tolist()
#     record['rectified_Face_selective_s_avg'] = (FSI_rect > .333).astype(np.float).mean(2).tolist()

#     features_shifted = features - np.array(features).min()
#     F_shifted = features_shifted[face_inds].mean(0)
#     BO_shifted = features_shifted[bo_inds].mean(0)
#     FSI_shifted = (F_shifted - BO_shifted) / (F_shifted + BO_shifted)
#     FSI_shifted_counts = [len((FSI_shifted > thres).nonzero()[0]) for thres in thresholds]
#     FSI_shifted_fractions = [c/ float(num_features) for c in FSI_shifted_counts]
#     record['shifted_fsi_fractions'] = FSI_shifted_fractions
#     record['shifted_FSI_s_avg'] = FSI_shifted.mean(2).tolist()
#     record['shifted_Face_selective_s_avg'] = (FSI_shifted > .333).astype(np.float).mean(2).tolist()
    
#     dprime = (F - BO) / features.std(0)
#     dprime_h, dprime_b = np.histogram(dprime, bins=50)
#     record['dprime_hist'] = dprime_h.tolist()
#     record['dprime_bins'] = dprime_b.tolist()
#     record['dprime_selective_fraction'] = float(len((dprime.flatten() > 1).nonzero()[0]) / float(num_features))
#     record['dprime_selective_s_avg'] = (dprime > 1).astype(np.float).mean(2).tolist()
    
#     Z = np.row_stack([features[face_inds], features[bo_inds]])
#     R = Z.argsort(0)
#     nf = len(face_inds)
#     ndist = len(bo_inds)
#     roc_FSI = (R[:nf].sum(0) - ((nf)**2 + nf)/2.) / float(nf*ndist)
#     roc_FSI_counts = [len((roc_FSI > thres).nonzero()[0]) for thres in thresholds]
#     roc_FSI_fractions = [c/ float(num_features) for c in roc_FSI_counts]
#     record['roc_fsi_fractions'] = roc_FSI_fractions
#     record['roc_FSI_s_avg'] = roc_FSI.mean(2).tolist()
#     record['roc_Face_selective_s_avg'] = (roc_FSI > .75).astype(np.float).mean(2).tolist()
    
#     if train:
#         features = features.reshape((fs[0],num_features))
#         STATS = ['train_accuracy','train_ap','train_auc','test_accuracy','test_ap','test_auc']
#         catfuncs = [('Face_Nonface',lambda x : x['name'] if x['name'] == 'Face' else 'Nonface'),
#                     ('Face_Body_Object',None)]
#         record['training_data'] = {}
#         for (problem, func) in catfuncs:
#             results = traintest(dataset, features, catfunc=func, **training_data)
#             stats = {}
#             for stat in STATS:
#                 stats[stat] = np.mean([r[1][stat] for r in results])
#             record['training_data'][problem] = stats
#         record['loss'] = 1 - (record['training_data']['Face_Nonface']['test_accuracy'])/100.
#     record['status'] = 'ok'
#     return record



########################
########Facelike bandits
########################

# def evaluate_facelike(config, credentials, FSI=None):
#     dataset = simffa_datasets.Facelike(credentials)
#     X, labels = dataset.img_regression_task()
#     all_paths, labels1 = dataset.raw_regression_task()
#     assert (labels == labels1).all()
#     assert (all_paths == sorted(all_paths)).all()
#     features = get_features(X, config, verbose=True)
#     fs = features.shape
#     num_features = fs[1]*fs[2]*fs[3]
#     if FSI is not None:
#         FSI_shape = FSI.shape
#         assert FSI_shape == fs[1:]
#         FSI = np.ravel(FSI)

#     record = {}
#     record['num_features'] = num_features
#     record['feature_shape'] = fs

#     subjects = [('subject_avg','avg')] # + [('subject_' + str(ind),ind) for ind in range(5)]

#     bins = np.arange(-1, 1, .01)
#     record['bins'] = bins.tolist()
#     for subject, judgement in subjects:
#         record[subject] = {}
#         for name, subset in dataset.SUBSETS + [('all',None)]:
#             print('Evaluating', subject, name)
#             subpaths, sublabels = dataset.raw_regression_task(subset=subset,
#                                                         judgement=judgement)
#             inds = all_paths.searchsorted(subpaths)
#             f_subset = features[inds]
#             label_subset = labels[inds]
#             P, P_prob = pearsonr(f_subset, label_subset)
#             P = np.ma.masked_array(P, np.isnan(P))
            
#             f_rshp = f_subset.reshape((fs[0],num_features))
#             P_pop, P_pop_prob = sp_stats.pearsonr(f_rshp.mean(1), label_subset)
#             S_pop, S_pop_prob = sp_stats.spearmanr(f_rshp.mean(1), label_subset)

#             data = {}
#             data['Pearson_hist'] = np.histogram(P, bins)[0].tolist()
#             data['Pearson_avg'] = float(P.mean())
#             data['Pearson_s_avg'] = P.mean(2).tolist()
#             data['Pearson_pop_avg'] = float(P_pop)
#             data['Spearman_pop_avg'] = float(S_pop)
            
#             if FSI is not None:
#                 assert FSI_shape == P.shape == f_subset.shape[1:]
#                 sel_inds = (FSI > 1./3)
#                 data['Pearson_pop_avg_sel'] = float(sp_stats.pearsonr(f_rshp[sel_inds].mean(1), label_subset))
#                 data['Spearman_pop_avg_sel'] = float(sp_stats.spearmanr(f_rshp[sel_inds].mean(1), label_subset))
#                 P = P.ravel()
#                 sel_inds = np.invert(np.isnan(P))
#                 data['Pearson_FSI_corr'] = np.corrcoef(FSI[sel_inds], P[sel_inds]).tolist()
#                 sel_inds = (FSI > 1./3) & np.invert(np.isnan(P))
#                 data['Pearson_hist_sel'] = np.histogram(P[sel_inds], bins)[0].tolist()
#                 data['Pearson_avg_sel'] = float(P[sel_inds].mean())
#                 data['Pearson_FSI_corr_sel'] = np.corrcoef(FSI[sel_inds], P[sel_inds]).tolist()
 
#             record[subject][name] = data

#     return record


                            
