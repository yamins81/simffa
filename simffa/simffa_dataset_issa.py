# -*- coding: utf-8 -*-

# Copyright (C) 2011
# Authors:  Elias Issa and Dan Yamins
# edited by Rishi Rajalingham (2013)
# entire imageset (4713 images) used for face patch recordings 
# + neural and psychophys data

# License: Simplified BSD

import os
import shutil
import hashlib
import cPickle

import numpy as np
from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader
from skdata.data_home import get_data_home

class MTData(object):

    def __init__(self, meta=None, seed=0, ntrain=100, ntest=100, num_splits=5):
        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits

        if meta is not None:
            self._meta = meta
        self.name = self.__class__.__name__
  
    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.name, *suffix_paths)

    def fetch(self, download_if_missing=True):
        home = self.home()

        if not download_if_missing:
            raise IOError("'%s' exists!" % home)

        # download archive
        url = self.URL
        sha1 = self.SHA1
        basename = os.path.basename(url)
        archive_filename = os.path.join(home, basename)
        if not os.path.exists(archive_filename):
            if not download_if_missing:
                return
            if not os.path.exists(home):
                os.makedirs(home)
            download(url, archive_filename, sha1=sha1)

        # extract it
        if not os.path.exists(self.home(self.SUBDIR)):
            extract(archive_filename, home, sha1=sha1, verbose=True)

    def clean_up(self):
        if os.path.isdir(self.home()):
            shutil.rmtree(self.home())

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
        im_oi = metadata[self.IM_OI]

        meta = []
        for ii in im_oi:
            i = int(ii)
            img_fn = os.path.join(self.home(self.SUBDIR), metadata['img_filenames'][i-1][0])
            img_data = open(img_fn, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()

            label1 = metadata['psy_face'][i-1]
            label2 = metadata['psy_eye'][i-1]
            label3 = metadata['psy_nose'][i-1]

            neural1 = metadata['neural_pl'][i-1,:]
            neural2 = metadata['neural_ml'][i-1,:]
            neural3 = metadata['neural_al'][i-1,:]
            
            data = dict(filename=img_fn,
                        id=i,
                        label1=label1,
                        label2=label2,
                        label3=label3,
                        neural1=neural1,
                        neural2=neural2,
                        neural3=neural3,
                        sha1=sha1)
            meta += [data]

        return meta

    @property
    def splits(self):
        if not hasattr(self, '_splits'):
            seed = self.seed
            ntrain = self.ntrain
            ntest = self.ntest

            num_splits = self.num_splits
            self._splits = self.get_regression_splits(seed, num_splits)
        return self._splits

    def get_regression_splits(self, seed, num_splits):
        x = np.array([ np.isnan(self.get_neural_labels(i+1)[:,0]) for i in range(3)])
        t = np.nonzero(np.logical_or(x[0,:], x[1,:], x[2,:]))
        t = np.array(t).T

        nIm = len(t)
        print nIm
        ntrain = int(0.75 * nIm)
        ntest = int(0.25 * nIm)

        rng = np.random.RandomState(seed)
        splits = {}
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []
            
            perm = rng.permutation(nIm)
            for ind in perm[:ntrain]:
                splits['train_' + str(split_id)].append(t[ind])
            for ind in perm[ntrain: ntrain + ntest]:
                splits['test_' + str(split_id)].append(t[ind])
        return splits

    def get_neural_labels(self, neural_id=1):
        label_name = 'neural' + str(neural_id)
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        labels = np.asarray([np.double(self.meta[ind][label_name]) for ind in inds])
        LABELS = np.array(labels)   
        return LABELS

    def get_labels(self, label_id=1):
        label_name = 'label' + str(label_id)
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        labels = np.asarray([self.meta[ind][label_name] for ind in inds])
        LABELS = np.array(labels)
        return LABELS

    def get_images(self, label_id=1):
        if label_id == None:
            label_id = 1
        label_name = 'label' + str(label_id)
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        image_paths = [self.meta[ind]['filename'] for ind in inds]
        imgs = larray.lmap(ImgLoader(ndim=2, shape=(400,400), dtype='uint8', mode='L'),
                           image_paths)
        labels = np.asarray([self.meta[ind][label_name] for ind in inds])
    
        IMGS = np.array(imgs)
        LABELS = np.array(labels)
   
        return IMGS, LABELS



class IssaMTData_408(MTData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_issa_mt.zip'
    SHA1 = 'd8c55f7fae905fa85be830706cf6d9c07832fa49'
    SUBDIR = 'DAT_issa_mt'
    IM_OI = 'img_oi_408'


class IssaMTData_818(MTData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_issa_mt.zip'
    SHA1 = 'd8c55f7fae905fa85be830706cf6d9c07832fa49'
    SUBDIR = 'DAT_issa_mt'
    IM_OI = 'img_oi_818'



# class MTData_March082013(MTData):
#     URL = 'http://dicarlocox-datasets.s3.amazonaws.com/simffa_dat.zip'
#     SHA1 = '1cb9e893d7a582040aef898d00f3d370bf274efe'
#     SUBDIR = 'DAT_mt'
#     IMG_fn = 'img_all.txt'
#     LABEL1_fn = 'psyFaceMag_20121012_210.txt'
#     LABEL2_fn = 'psyEyeMag_20121228_285.txt'
#     LABEL3_fn = 'psyNoseMag_20130201_285.txt'
#     # IMG_OI_fn = 'img_oi818.txt'
#     IMG_OI_fn = 'img_oi408.txt'
#     PL_fn = 'neuralPLpop.txt'
#     ML_fn = 'neuralMLpop.txt'
#     AL_fn = 'neuralALpop.txt'


# class MTData_Feb222013(MTData):
#     URL = './'
#     SHA1 = '088387e08ac008a0b8326e7dec1f0a667c8b71d0'
#     SUBDIR = 'DAT_mt'
#     IMG_fn = 'img_all.txt'
#     LABEL1_fn = 'psyFaceMag_20121012_210.txt'
#     LABEL2_fn = 'psyEyeMag_20121228_285.txt'
#     LABEL3_fn = 'psyNoseMag_20130201_285.txt'
#     # IMG_OI_fn = 'img_oi818.txt'
#     IMG_OI_fn = 'img_oi408.txt'

