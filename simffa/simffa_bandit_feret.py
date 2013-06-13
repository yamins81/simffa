
import numpy as np
import scipy.stats as sp_stats
from yamutils.stats import pearsonr, spearmanr
from dldata_classifier import train_scikits

from pyll import scope
import hyperopt
from hyperopt import base
import simffa_params as sp
from simffa_utils import get_features

import simffa.simffa_dataset_feret as feret

def get_FERET_classification(features, labels, splits):

    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])

    num_splits = int(len(splits)/2)
    id_test_accuracy = 0
    view_test_rsq = 0
    for i in range(num_splits):
        train_inds = np.array(splits['train_' + str(i)])
        test_inds = np.array(splits['test_' + str(i)])
        train_X = features[train_inds]
        test_X = features[test_inds]

        # id classification
        train_y = labels[train_inds,0]
        test_y = labels[test_inds,0]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'libSVM')
        test_accuracy = test_accuracy + result[1]['test_accuracy']
        print 'done id classification'

        #viewpoint regression
        train_y = labels[train_inds,1]
        test_y = labels[test_inds,1]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression')
        view_test_rsq = view_test_rsq + result[1]['test_rsquared']
        print 'done vp regression'

    test_accuracy = test_accuracy / np.double(num_splits)
    view_test_rsq = view_test_rsq / np.double(num_splits)
    return test_accuracy, view_test_rsq

@scope.define
def feret_bandit_evaluate(config=None):
    dataset = feret.FERET()
    imgs,labels = dataset.get_images()
    splits = dataset.splits()

    nIm = labels.shape[0]
    nLabels = labels.shape[1]
    features = get_features(imgs, config, verbose=False)
    
    print 'Evaluating model on id classificaiton and viewpoint regression'
    test_accuracy, view_test_rsq = get_FERET_classification(features, labels, splits)
    results = {}
    results['id_accuracy'] = test_accuracy
    results['view_rsq'] = view_test_rsq

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - (view_test_rsq + test_accuracy)/2
    record['status'] = 'ok'
    return record


@base.as_bandit()
def Simffa_FeretTask_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.feret_bandit_evaluate(template)

@base.as_bandit()
def Simffa_FeretTask_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.feret_bandit_evaluate(template)

@base.as_bandit()
def Simffa_FeretTask_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.feret_bandit_evaluate(template)


                            
