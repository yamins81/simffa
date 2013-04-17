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

import numpy as np
import tabular as tb
import Image
import random

import hyperopt
from hyperopt import base
from pyll import scope

import simffa_params as sp
import simffa_mtDat as mtdat
from simffa_utils import slm_h5

import skdata as skd
from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader

from dldata_classifier import train_scikits

class FeretData(object):

    def __init__(self, meta=None, seed=0, ntrain=100, ntest=100, num_splits=5):
        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits

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

        img_filenames = sorted(os.listdir(self.home(self.SUBDIR)))
        img_filenames = [os.path.join(self.home(self.SUBDIR),x) for x in img_filenames]
        pose_defs = {'ba':0, 'bb':50, 'bc':40, 'bd':25, 'be':15, 'fa':0, 'fb':0,
                    'ql':-22.5, 'qr':22.5, 'hl':-67.5, 'hr':67.5, 'pl':-90,'pr':90}  
        meta = []
        for img_filename in img_filenames:
            img_data = open(img_filename, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            subject_id = int(os.path.split(img_filename)[1].split('.')[0][:5])
            pose_tag = (os.path.split(img_filename)[1].split('.')[0][5:7])
            pose_id = pose_defs[pose_tag]
            data = dict(filename=img_filename,
                        subject_id=subject_id,
                        pose_id=pose_id,
                        sha1=sha1)

            meta += [data]

        return meta

    def home(self, *suffix_paths):
        return path.join('/home/data/', *suffix_paths)

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

    def get_images(self):
        
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        image_paths = [self.meta[ind]['filename'] for ind in inds]
        imgs = larray.lmap(ImgLoader(ndim=2, dtype='uint8', mode='L'), image_paths)
        labels = np.asarray([ (self.meta[ind]['subject_id'], self.meta[ind]['pose_id']) for ind in inds])
    
        IMGS = np.array(imgs)
        LABELS = np.array(labels)
   
        return IMGS, LABELS

class FERET(FeretData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FERET.zip'
    SHA1 = '3666828effe1cc77b6297406f1e0f9799137068e'
    SUBDIR = 'FERET'

class FERET_gray(FeretData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FERET_gray.zip'
    SHA1 = '9c694eb2ae49022566c641a04e5eea776928869a'
    SUBDIR = 'FERET_gray'

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

def getFERETsplits(labels, seed, nClasses, num_splits):
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
            for ind in perm[:1]:
                splits['train_' + str(split_id)].append(ind)
            for ind in perm[1:]:
                splits['test_' + str(split_id)].append(ind)
    return splits

def get_FERET_classification(features, labels, nClasses=36, num_splits=2):

    splits = getFERETsplits(labels, np.random.randint(nClasses), nClasses, num_splits)
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

        #viewpoint regression
        train_y = labels[train_inds,1]
        test_y = labels[test_inds,1]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'pls.PLSRegression')
        view_test_rsq = view_test_rsq + result[1]['test_rsquared']

    test_accuracy = test_accuracy / num_splits
    view_test_rsq = view_test_rsq / num_splits
    return test_accuracy, view_test_rsq

@scope.define
def feret_bandit_evaluate(config=None):
    dataset = FERET()
    imgs,labels = dataset.get_images()
    nIm = labels.shape[0]
    print 'Loading ' + str(nIm) + ' FERET imgs...'
    features = get_features(imgs, config, verbose=False)
    
    print 'Evaluating model on id classificaiton and viewpoint regression'
    test_accuracy, view_test_rsq = get_FERET_classification(features, labels, nClasses=36, num_splits=2)
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

@scope.define
def feret_bandit_evaluate2(config=None, stats=None):
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
        tmp = [get_regression_results(features, curr_neural[:,ei]) for ei in range(4)]
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
def FERETL3Bandit(template=None):
    if template==None:
        template = sp.l3_params
    return scope.feret_bandit_evaluate(template)

@base.as_bandit()
def FERETL2Bandit(template=None):
    if template==None:
        template = sp.l2_params
    return scope.feret_bandit_evaluate(template)

