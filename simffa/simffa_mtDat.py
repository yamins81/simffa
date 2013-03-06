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
import Image
import random
import skdata as skd
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader

def get_transformed_image(img):
    cimg = Image.fromarray(img).copy()
    if np.random.random() > 0.5:
        cimg = cimg.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() > 0.5:
        cimg = cimg.transpose(Image.FLIP_TOP_BOTTOM) 
    cimg = cimg.rotate(360*np.random.random())

    xoffset = np.int(np.random.random()*200);
    yoffset = np.int(np.random.random()*200);
    cimg = cimg.offset(xoffset, yoffset)
    
    return cimg
  
# base class - MTurk dataset
class MTData(object):

    def __init__(self, meta=None, seed=0, ntrain=100, ntest=100, num_splits=5):

        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits
        self.names = ['Face','Body','Object']

        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__
  
    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):

        # all image filenames
        f = open(self.IMG_fn, 'r')
        img_filenames = [[x for x in line.split()] for line in f]
        f.close()
        img_filenames = np.array(img_filenames)
        
        # all psychophysical labels
        f = open(self.LABEL_fn, 'r')
        label_data  = [[np.double(x) for x in line.split()] for line in f]
        f.close()

        # indices of images of interest 
        f = open(self.IMG_OI_fn, 'r')
        img_oi  = [[int(x) for x in line.split()] for line in f]
        f.close()
        img_oi = np.double(img_oi)

        meta = []

        s_im = img_oi.shape
        maxImgs = 1000
        nIm = min(maxImgs, s_im[0])
        for i in range(nIm):
            ind = np.int(img_oi[i])
            img_filename = img_filenames[ind-1][0]
            name = label_data[ind-1][1]
            img_data = open(img_filename, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()

            data = dict(name=name,
                        id=ind,
                        filename=img_filename,
                        sha1=sha1)
            meta += [data]

        return meta

    @property
    def splits(self):
        """
        generates splits and attaches them in the "splits" attribute
        """
        if not hasattr(self, '_splits'):
            seed = self.seed
            ntrain = self.ntrain
            ntest = self.ntest
            num_splits = self.num_splits
            self._splits = self.generate_splits(seed, ntrain,
                                                ntest, num_splits)
        return self._splits

    def generate_splits(self, seed, ntrain, ntest, num_splits, labelset=None, catfunc=None):
        meta = self.meta
        if labelset is not None:
            assert catfunc is not None
        else:
            labelset = self.names
            catfunc = lambda x : x['name']

        ntrain = self.ntrain
        ntest = self.ntest
        rng = np.random.RandomState(seed)
        splits = {}
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []
            for label in labelset:
                cat = [m for m in meta if catfunc(m) == label]
                L = len(cat)

                assert L >= ntrain + ntest, 'category %s too small' % name
                perm = rng.permutation(L)
                for ind in perm[:ntrain]:
                    splits['train_' + str(split_id)].append(cat[ind]['filename'])
                for ind in perm[ntrain: ntrain + ntest]:
                    splits['test_' + str(split_id)].append(cat[ind]['filename'])
        return splits


    def generate_regression_splits(self, labels, seed, ntrain, ntest, num_splits):
        meta = self.meta
        nIm = np.array(meta).shape[0]
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

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Standard Tasks
    # ------------------------------------------------------------------------

    def raw_classification_task(self, split=None):
        """Return image_paths, labels"""
        if split:
            inds = self.splits[split]
        else:
            inds = xrange(len(self.meta))
        image_paths = [self.meta[ind]['filename'] for ind in inds]
        names = np.asarray([self.meta[ind]['name'] for ind in inds])
        # labels = int_labels(names)
        labels = names
        return image_paths, labels

    def img_classification_task(self, dtype='uint8', split=None):
        img_paths, labels = self.raw_classification_task(split=split)
        imgs = skd.larray.lmap(ImgLoader(ndim=2, shape=(400,400), dtype=dtype, mode='L'),
                           img_paths)
        return imgs, labels

    def invariant_img_classification_task(self, num_reps):
        imgs, labels = self.img_classification_task()
        IMGS = np.array(imgs).tolist()
        LABELS = labels.tolist()
        
        fs = imgs.shape

        for reps in range(num_reps):
            for i in range(fs[0]):
                img = get_transformed_image(imgs[i][:][:])
                label = labels[i]
                img = np.array(img).tolist()
                IMGS.append(img)
                LABELS.append(label)

        IMGS = np.array(IMGS)
        LABELS = np.array(LABELS)
        return IMGS, LABELS


class MTData_Feb222013(MTData):
    URL = './'
    SHA1 = '088387e08ac008a0b8326e7dec1f0a667c8b71d0'
    SUBDIR = 'DAT_mt'
    IMG_fn = 'img_all.txt'
    LABEL_fn = 'psyFaceMag_20121012_210.txt'
    # IMG_OI_fn = 'img_oi818.txt'
    IMG_OI_fn = 'img_oi408.txt'



