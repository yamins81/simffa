
import numpy as np
from dldata_classifier import train_scikits
from pyll import scope
from hyperopt import base
import simffa_params as sp
from simffa_utils import get_features
import simffa_dataset_issa as issa

def get_regression_result(features, labels, splits):
    
    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])
    
    rsq = 0;
    num_splits = int(len(splits)/2)
    for ind in range(num_splits):
        train_inds = np.array(splits['train_' + str(ind)])
        test_inds = np.array(splits['test_' + str(ind)])
        train_X = features[train_inds]
        test_X = features[test_inds]
        train_y = labels[train_inds]
        test_y = labels[test_inds]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression', regression=True)
        rsq = rsq+result[1]['test_rsquared']
    rsq = rsq / num_splits

    return rsq

@scope.define
def issa_evaluate_neuralComparisons(config=None):
    dataset = issa.IssaMTData_818()
    imgs,labels = dataset.get_images()
    splits = dataset.splits()
    features = get_features(imgs, config, verbose=False)

    nIm = labels.shape[0]
    fs = features.shape

    print 'regressing model on neural labels'
    regress_r2 = {}
    neuralname = ['pl', 'ml', 'al']

    for neural_i in range(len(neuralname)):
        curr_neural = dataset.get_neural_labels(neural_i+1)
        tmp = np.array([sb.get_regression_result(features, curr_neural[:,ei], splits) for ei in range(4)])
        regress_r2[neuralname[neural_i]] = tmp

    results = {}
    results['regress_r2'] = regress_r2
    results['fs'] = fs

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 0
    record['status'] = 'ok'
    return record


@scope.define
def issa_evaluate_psychComparisons(config=None):
    dataset = issa.IssaMTData_818()
    imgs,labels = dataset.get_images()
    splits = dataset.splits()
    features = get_features(imgs, config, verbose=False)

    nIm = labels.shape[0]
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    
    print 'regressing model on psych labels'
    regress_r2 = {}
    labelnames = ['face', 'eye']

    for label_i in range(len(labelnames)):
        label = dataset.get_labels(label_i+1)
        regress_r2[labelnames[label_i]] = get_regression_result(features, label, splits)
        print 'done ' + labelnames[label_i]
    
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
def issa_evaluate_allComparisons(config=None):
    dataset = issa.IssaMTData_818()
    imgs,labels = dataset.get_images(label_id)
    splits = dataset.splits()
    features = get_features(imgs, config, verbose=False)

    nIm = labels.shape[0]
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    
    print 'regressing model on psych + neural labels'
    regress_r2 = {}
    labelnames = ['face', 'eye']
    neuralname = ['pl', 'ml', 'al']

    for label_i in range(len(labelnames)):
        label = dataset.get_labels(label_i+1)
        regress_r2[labelnames[label_i]] = get_regression_result(features, label, splits)
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

# @scope.define
# def issa_evaluate_patches(config=None):
#     dataset = issa.IssaMTData_408()
#     imgs,labels = dataset.get_images(1)
#     nIm = labels.shape[0]
#     print 'Loading ' + str(nIm) + 'imgs..'
#     features = get_features(imgs, config, verbose=False)

#     shuf = False
#     if shuf==True:
#         t = np.random.random_integers(0,len(labels)-1,len(labels))
#         labels = labels[t]

#     fs = features.shape
#     num_features = fs[1]*fs[2]*fs[3]    
#     record = {}
#     record['spec'] = config
#     record['num_features'] = num_features
#     record['feature_shape'] = fs

#     print 'computing regression error'
#     num_splits = 4
#     num_train = np.floor(0.75*nIm)
#     num_test = np.floor(0.10*nIm)
#     seed = np.random.randint(num_splits)
#     shuf_ind = np.random.random_integers(0,len(labels)-1,len(labels))
#     psy_rsq_mu = regression_traintest(features, labels, seed, num_train, num_test, num_splits)
#     psy_rsq_mu_shuf = regression_traintest(features, labels[shuf_ind], seed, num_train, num_test, num_splits)
    
#     # features = features.mean(3)
#     # print 'computed map'
#     # psyCorr = sanal.getPearsonCorr2D(features, labels)
#     # nonnanCorr = np.array(psyCorr).ravel()
#     # nonnanCorr = nonnanCorr[~np.isnan(nonnanCorr)]
#     # hist_n, hist_x = np.histogram(nonnanCorr, bins=50)

#     print 'storing stats'
#     results = {}
#     results['psyRegress'] = psy_rsq_mu
#     results['psyRegress_shuf'] = psy_rsq_mu_shuf

#     # results['psyCorr'] = psyCorr
#     # results['psyCorr_hist_n'] = hist_n.tolist()
#     # results['psyCorr_hist_x'] = hist_x[1:].tolist()
#     # results['psyCorr_mu'] = np.abs(psyCorr).ravel().mean(0)
#     # results['psyCorr_cluster'] = sanal.getClusterSize(psyCorr)
#     # results['psyCorr_blob'] = sanal.getBlobiness(psyCorr)
#     # results['psyCorr_topog'] = sanal.getTopographicOrg(psyCorr)

#     record['results'] = results
#     record['attachments'] = {}
#     record['loss'] = 1 - psy_rsq_mu
#     record['status'] = 'ok'
#     return record


# neural fits
@base.as_bandit()
def Simffa_IssaNeural_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.issa_evaluate_neuralComparisons(template)

@base.as_bandit()
def Simffa_IssaNeural_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.issa_evaluate_neuralComparisons(template)

@base.as_bandit()
def Simffa_IssaNeural_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.issa_evaluate_neuralComparisons(template)

# psych fits
@base.as_bandit()
def Simffa_IssaPsych_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.issa_evaluate_psychComparisons(template)

@base.as_bandit()
def Simffa_IssaPsych_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.issa_evaluate_psychComparisons(template)

@base.as_bandit()
def Simffa_IssaPsych_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.issa_evaluate_psychComparisons(template)

    
# both fits
@base.as_bandit()
def Simffa_IssaGeneral_Bandit(template=None):
    if template==None:
        template = sp.l3_params
    return scope.issa_evaluate_allComparisons(template)

                            
