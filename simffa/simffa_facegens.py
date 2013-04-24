# -*- coding: utf-8 -*-

# Copyright (C) 2011
# Authors:  Elias Issa and Dan Yamins
# edited by Rishi Rajalingham (2013)

# License: Simplified BSD

import os
from os import path
import shutil
from glob import glob
import hashlib
import cPickle


import numpy as np
import tabular as tb
import Image
import random

import hyperopt
from hyperopt import base
from pyll import scope

import simffa_params as sp
import simffa_bandit as sb
import simffa_mtDat as mtdat
import skdata as skd

from simffa_utils import slm_h5
from dldata_classifier import train_scikits

from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader

class FaceGenData(object):

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta
        self.name = self.__class__.__name__
  
    @property
    def meta(self):            
        if not hasattr(self, '_meta'):
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        home = self.home()

        meta_fn = os.path.join(self.home(self.SUBDIR),'metadata.pkl') 
        file = open(meta_fn,'r')
        metadata = cPickle.load(file)
        meta = []
        for mi in range(len(metadata)):
            img_id = metadata['id'][mi]

            subject_id = int(img_id[4:7])
            if subject_id < 4:
                continue

            express_id = int(img_id[8:-1])

            img_fn = os.path.join(self.home(self.SUBDIR),img_id + '.png') 
            img_data = open(img_fn, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            
            ty = metadata['ty'][mi][0]
            tz = metadata['tz'][mi][0]
            s = metadata['s'][mi][0]
            rxy = metadata['rxy'][mi][0]
            rxz = metadata['rxz'][mi][0]
            ryz = metadata['ryz'][mi][0]

            data = dict(filename=img_fn,
                        subject_id=subject_id,
                        express_id=express_id,
                        ty=ty,tz=tz,s=s,rxy=rxy,rxz=rxz,ryz=ryz,
                        sha1=sha1)
            meta += [data]

        return meta

    def home(self, *suffix_paths):
        return path.join('/home/data/', *suffix_paths)
        # return path.join('/mindhive/dicarlolab/u/rishir/simffa/simffa/data_tmp', *suffix_paths)

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""

        home = self.home()

        if not download_if_missing:
            raise IOError("'%s' exists!" % home)

        # download archive
        url = self.URL
        sha1 = self.SHA1
        basename = path.basename(url)
        archive_filename = path.join(home, basename)
        if not path.exists(archive_filename):
            if not download_if_missing:
                return
            if not path.exists(home):
                os.makedirs(home)
            download(url, archive_filename, sha1=sha1)

        # extract it
        if not path.exists(self.home(self.SUBDIR)):
            extract(archive_filename, home, sha1=sha1, verbose=True)

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    def get_label(self, ind):
        label = [self.meta[ind]['subject_id'], self.meta[ind]['express_id'], 
            self.meta[ind]['s'], self.meta[ind]['ty'],self.meta[ind]['tz'],
            self.meta[ind]['rxy'], self.meta[ind]['rxz'],self.meta[ind]['ryz']]
        return label

    def get_images(self):
        
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        image_paths = [self.meta[ind]['filename'] for ind in inds]
        imgs = larray.lmap(ImgLoader(ndim=2, dtype='uint8', mode='L'), image_paths)
        labels = np.asarray([ (self.meta[ind]['subject_id'], self.meta[ind]['express_id'], 
            self.meta[ind]['s'], self.meta[ind]['ty'],self.meta[ind]['tz'],
            self.meta[ind]['rxy'], self.meta[ind]['rxz'],self.meta[ind]['ryz']) for ind in inds])
    
        IMGS = np.array(imgs)
        LABELS = np.array(labels)
   
        return IMGS, LABELS

class FaceGen_small(FaceGenData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_fg_s.zip'
    SHA1 = 'b97d81032b9ebc07e2612730be2ba701a19aa307'
    SUBDIR = 'DAT_fg_s'

## methods for bandit ##

def get_features(X, config, verbose=False, dirname=None, fname_tag=None):
    features = slm_h5(
                    desc=config['desc'],
                    X=X,
                    basedir=dirname,
                    name=fname_tag, 
                    save=False) 
    features = np.array(features)
    return features

def get_splits(labels, seed, nClasses, num_splits):
    label_id = 0
    uniqueLabels = np.unique(labels[:,label_id])
    nIm = np.array(labels).shape[0]
    rng = np.random.RandomState(seed)
    splits = {}

    for split_id in range(num_splits):
        splits['train_' + str(split_id)] = []
        splits['test_' + str(split_id)] = []

        for i in range(nClasses):
            iMlbl_i = [j for j in range(nIm) if labels[j,label_id] == uniqueLabels[i]]
            perm_i = rng.permutation(len(iMlbl_i))
            perm = [iMlbl_i[p_i] for p_i in perm_i]
            for ind in perm[:len(perm)/2]:
                splits['train_' + str(split_id)].append(ind)
            for ind in perm[len(perm)/2 :]:
                splits['test_' + str(split_id)].append(ind)
    return splits

def get_regression_result(labels, train_X, test_X, train_inds, test_inds):
    train_y = labels[train_inds]
    test_y = labels[test_inds]
    train_Xy = (train_X, train_y)
    test_Xy = (test_X, test_y)
    result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression')
    test_rsq = result[1]['test_rsquared']
    return test_rsq

def get_classification_result(labels, train_X, test_X, train_inds, test_inds):
    train_y = labels[train_inds]
    test_y = labels[test_inds]
    train_Xy = (train_X, train_y)
    test_Xy = (test_X, test_y)
    result = train_scikits(train_Xy, test_Xy, 'libSVM')
    test_accuracy = result[1]['test_accuracy'] / 100
    return test_accuracy


def evaluate_on_tasks(features, labels, nClasses=36, num_splits=2):

    splits = get_splits(labels, np.random.randint(nClasses), nClasses, num_splits)
    fs = features.shape
    if np.array(fs).shape[0] == 4:
        features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])

    id_test_accuracy = 0
    express_test_accuracy = 0
    
    pose_regress = {}
    pose_params = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz']
    view_test_rsq = 0
    for i in range(num_splits):
        train_inds = np.array(splits['train_' + str(i)])
        test_inds = np.array(splits['test_' + str(i)])
        train_X = features[train_inds]
        test_X = features[test_inds]

        # id classification
        id_test_accuracy = id_test_accuracy + get_classification_result(labels[:,0], train_X, test_X, train_inds, test_inds)
        # train_y = labels[train_inds,0]
        # test_y = labels[test_inds,0]
        # train_Xy = (train_X, train_y)
        # test_Xy = (test_X, test_y)
        # result = train_scikits(train_Xy, test_Xy, 'libSVM')
        # id_test_accuracy = id_test_accuracy + result[1]['test_accuracy']
        print 'done id classification'

        express_test_accuracy = express_test_accuracy + get_classification_result(labels[:,1], train_X, test_X, train_inds, test_inds)
        # train_y = labels[train_inds,1]
        # test_y = labels[test_inds,1]
        # train_Xy = (train_X, train_y)
        # test_Xy = (test_X, test_y)
        # result = train_scikits(train_Xy, test_Xy, 'libSVM')
        # express_test_accuracy = express_test_accuracy + result[1]['test_accuracy']
        print 'done expression classification'

        for li in range(len(pose_params)):
            pose_regress[pose_params[li]] = pose_regress[pose_params[li]] + get_regression_result(labels[:,2+li], train_X, test_X, train_inds, test_inds)
        #viewpoint regression
        # train_y = labels[train_inds,1]
        # test_y = labels[test_inds,1]
        # train_Xy = (train_X, train_y)
        # test_Xy = (test_X, test_y)
        # result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression')
        # view_test_rsq = view_test_rsq + result[1]['test_rsquared']
        # print 'done vp regression'

    id_test_accuracy = id_test_accuracy / num_splits
    express_test_accuracy = express_test_accuracy / num_splits

    avg_pose_regress = 0
    for li in range(len(pose_params)):
        pose_regress[pose_params[li]] = pose_regress[pose_params[li]] / num_splits
        avg_pose_regress = avg_pose_regress + (pose_regress[pose_params[li]])/len(pose_params)
    
    # print 'identity classification: ' + str(id_test_accuracy)
    # print 'expression classification: ' + str(express_test_accuracy)


    return id_test_accuracy, express_test_accuracy, pose_regress, avg_pose_regress

@scope.define
def fgs_bandit_evaluate(config=None):
    dataset = FaceGen_small()
    imgs,labels = dataset.get_images()
    nIm = labels.shape[0]
    nLabels = labels.shape[1]
    print 'Loading ' + str(nIm) + ' facegen imgs...'
    print 'with ' + str(nLabels) + ' labels...'
    features = get_features(imgs, config, verbose=False)
    
    print 'Evaluating model on id classificaiton and viewpoint regression'
    accuracy1, accuracy2, accuracy3, avg_accuracy3 = evaluate_on_tasks(features, labels, nClasses=7, num_splits=10)
    results = {}
    results['id_accuracy'] = accuracy1
    results['express_accuracy'] = accuracy2
    results['pose_accuracy'] = accuracy3
    results['avg_pose_accuracy'] = avg_accuracy3

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - (accuracy1 + accuracy2 + avg_accuracy3)/3
    record['status'] = 'ok'
    return record

@scope.define
def bandit_evaluate2(config=None, stats=None):
    dataset = mtdat.MTData_March082013()
    imgs,labels = dataset.get_images(label_id)
    nIm = labels.shape[0]
    print 'Loading ' + str(nIm) + 'MTURK imgs...'
    features = get_features(imgs, config, verbose=False)

    print 'regressing model on neural labels'
    regress_r2 = {}
    neuralname = ['pl', 'ml', 'al']

    for neural_i in range(len(neuralname)):
        curr_neural = dataset.get_neural_labels(neural_i+1)
        tmp = [sb.get_regression_results(features, curr_neural[:,ei]) for ei in range(4)]
        regress_r2[neuralname[neural_i]] = tmp

    results = {}
    results['regress_r2'] = regress_r2
    results['feret_stats'] = stats

    record = {}
    record['spec'] = config
    record['results'] = results
    record['attachments'] = {}
    record['loss'] = 1 - (view_test_rsq + test_accuracy)/2
    record['status'] = 'ok'
    return record

## bandits ##
@base.as_bandit()
def Simffa_FaceGen_L3Bandit(template=None):
    if template==None:
        template = sp.l3_params
    return scope.fgs_bandit_evaluate(template)

@base.as_bandit()
def Simffa_FaceGen_L2Bandit(template=None):
    if template==None:
        template = sp.l2_params
    return scope.fgs_bandit_evaluate(template)

