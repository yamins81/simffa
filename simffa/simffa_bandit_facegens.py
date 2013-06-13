
import numpy as np
from dldata_classifier import train_scikits
from pyll import scope
from hyperopt import base
import simffa_params as sp
from simffa_utils import get_features
import simffa_dataset_facegens as fgs

def get_regression_result(labels, train_X, test_X, train_inds, test_inds):
    train_y = labels[train_inds]
    test_y = labels[test_inds]
    train_Xy = (train_X, train_y)
    test_Xy = (test_X, test_y)
    result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression', regression=True)
    test_rsq = np.double(result[1]['test_rsquared'])
    return test_rsq

def get_classification_result(labels, train_X, test_X, train_inds, test_inds):
    train_y = labels[train_inds]
    test_y = labels[test_inds]
    train_Xy = (train_X, train_y)
    test_Xy = (test_X, test_y)
    result = train_scikits(train_Xy, test_Xy, 'libSVM')
    test_accuracy = np.double(result[1]['test_accuracy'] / 100.0)
    return test_accuracy

def evaluate_on_tasks(features, labels, splits):
    fs = np.array(features.shape)
    if fs.shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])

    id_test_accuracy = 0
    express_test_accuracy = 0
    pose_regress = 0

    num_splits = int(len(splits)/2)
    for i in range(num_splits):
        train_inds = np.array(splits['train_' + str(i)])
        test_inds = np.array(splits['test_' + str(i)])
        train_X = features[train_inds]
        test_X = features[test_inds]

        id_test_accuracy = id_test_accuracy + get_classification_result(labels[:,0], train_X, test_X, train_inds, test_inds)
        express_test_accuracy = express_test_accuracy + get_classification_result(labels[:,1], train_X, test_X, train_inds, test_inds)
        pose_regress = pose_regress + get_regression_result(labels[:,6], train_X, test_X, train_inds, test_inds)

    id_test_accuracy = id_test_accuracy / num_splits
    express_test_accuracy = express_test_accuracy / num_splits
    pose_regress = pose_regress / num_splits
    
    print '***'
    print 'identity classification: ' + str(id_test_accuracy)
    print 'expression classification: ' + str(express_test_accuracy)
    print ' pose regress' + str(pose_regress)
    print '***'
    return id_test_accuracy, express_test_accuracy, pose_regress


# evaluate models on a battery of face tasks (id, expression, pose)
@scope.define
def fgs_evaluate_faceTasks(config=None):

    dataset = fgs.FaceGen_small_var0()
    imgs,labels = dataset.get_images()
    splits = dataset.splits
    features = get_features(imgs, config, verbose=False)
    
    nIm = labels.shape[0]
    nLabels = labels.shape[1]
    fs = features.shape
    
    accuracy1, accuracy2, accuracy3 = evaluate_on_tasks(features, labels, splits)
    results = {}
    results['id_accuracy'] = accuracy1
    results['express_accuracy'] = accuracy2
    results['pose_accuracy'] = accuracy3
    results['fs'] = fs

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - (accuracy1 + accuracy2 + accuracy3)/3.0
    record['status'] = 'ok'

    return record

@scope.define
def fgs_evaluate_spaceMap(config=None):
    
    # dataset = FaceGen_small_var0()
    # imgs,labels = dataset.get_images()
    # nIm = labels.shape[0]
    # nLabels = labels.shape[1]
    # print 'Loading ' + str(nIm) + ' facegen imgs...'
    # print 'with ' + str(nLabels) + ' labels...'
    # features = get_features(imgs, config, verbose=False)
    
    # print 'Evaluating model on id classificaiton and viewpoint regression'
    # maps = mapToSpace(features, labels)
    results = {}
    # results['maps'] = maps
    
    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 0
    record['status'] = 'ok'
    return record



## bandits ##
"""" face tasks """
@base.as_bandit()
def Simffa_FaceGenTasks_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.fgs_evaluate_faceTasks(template)

@base.as_bandit()
def Simffa_FaceGenTasks_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.fgs_evaluate_faceTasks(template)

@base.as_bandit()
def Simffa_FaceGenTasks_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.fgs_evaluate_faceTasks(template)


# """ compare to neurons """
# @base.as_bandit()
# def Simffa_Issa_Bandit_v2(template=None):
#     if template==None:
#         template = sp.l3_params
#     return scope.fgs_bandit_evaluate2(template)

""" map to space """
@base.as_bandit()
def Simffa_FaceGenMap_Bandit_V1(template=None):
    if template==None:
        template = sp.v1like_params
    return scope.fgs_evaluate_faceTasks(template)

@base.as_bandit()
def Simffa_FaceGenMap_Bandit_L2(template=None):
    if template==None:
        template = sp.l2_params
    return scope.fgs_evaluate_faceTasks(template)
    
@base.as_bandit()
def Simffa_FaceGenMap_Bandit_L3(template=None):
    if template==None:
        template = sp.l3_params
    return scope.fgs_evaluate_faceTasks(template)


# """ compute fsi"""
# @base.as_bandit()
# def Simffa_Issa_Bandit_v4(template=None):
#     if template==None:
#         template = sp.l3_params
#     return scope.fgs_bandit_evaluate2(template)

                            
