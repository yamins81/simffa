# -*- coding: utf-8 -*-
"""Face-like judgements dataset.
"""

# Copyright (C) 2011
# Authors:  Elias Issa and Dan Yamins

# License: Simplified BSD


import os
from os import path
import shutil
from glob import glob
import hashlib

import numpy as np
from scipy.io import loadmat

import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils import download_boto, extract, int_labels
from skdata.utils.image import ImgLoader


class BaseFacelike(object):

    def __init__(self, credentials, meta=None, seed=0, ntrain=10, ntest=10, num_splits=5):
        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits
        self.credentials = credentials

        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'))
            self._get_meta = mem.cache(self._get_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

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
            download_boto(url, self.credentials, archive_filename, sha1=sha1)

        # extract it
        if not path.exists(self.home(self.SUBDIR)):
            extract(archive_filename, home, sha1=sha1, verbose=True)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        imdir = os.path.join(self.home(self.SUBDIR), 'ims')
        img_filenames = sorted(os.listdir(imdir))
        img_filenames = [os.path.join(imdir,x) for x in img_filenames]
        matfile = os.path.join(self.home(self.SUBDIR), 'config_psychophys.mat')
        matobj = loadmat(matfile)
        imnums = matobj['imnums'][:,0].tolist()
        imratings = matobj['imratings'].tolist()
        imratingdict = dict(zip(imnums,imratings))
        meta = []
        for img_filename in img_filenames:
            img_data = open(img_filename, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            ind = int(os.path.split(img_filename)[1].split('.')[0][2:])
            ratings = imratingdict[ind]

            data = dict(id=ind,
                        filename=img_filename,
                        ratings=ratings,
                        avg_rating=np.mean(ratings),
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

    def generate_splits(self, seed, ntrain, ntest, num_splits):
        meta = self.meta
        ntrain = self.ntrain
        ntest = self.ntest
        rng = np.random.RandomState(seed)
        splits = {}
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []
            L = len(meta)
            assert L >= ntrain + ntest, 'dataset too small for  %d-%d splits' % (ntrain, ntest)
            perm = rng.permutation(L)
            for ind in perm[:ntrain]:
                splits['train_' + str(split_id)].append(meta[ind]['filename'])
            for ind in perm[ntrain: ntrain + ntest]:
                splits['test_' + str(split_id)].append(meta[ind]['filename'])
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

    def labels(self, split = None, judgement='avg'):
        if split:
            inds = self.splits[split]
        else:
            inds = xrange(len(self.meta))
        if judgement == 'avg':
            labels = np.asarray([self.meta[ind]['avg_rating'] for ind in inds])
        else:
            labels = np.asarray([self.meta[ind]['rating'][judgement] for ind in inds])
        return labels
            
    def raw_regression_task(self, split=None, subset=None, judgement='avg'):
        """Return image_paths, labels"""
        if split:
            inds = self.splits[split]
        else:
            inds = np.array(range(len(self.meta)))
        image_paths = np.array([self.meta[ind]['filename'] for ind in inds])
        if subset is not None:
            subset = map(str,subset)
            image_ids = np.array([os.path.split(p)[-1].split('.')[0][2:] for p in image_paths])
            sub_inds = np.searchsorted(image_ids, subset)
            inds = inds[sub_inds]
            image_paths = image_paths[sub_inds]
        if judgement == 'avg':
            labels = np.asarray([self.meta[ind]['avg_rating'] for ind in inds])
        else:
            labels = np.asarray([self.meta[ind]['rating'][judgement] for ind in inds])
        return image_paths, labels

    def img_regression_task(self, dtype='uint8', split=None, judgement='avg'):
        img_paths, labels = self.raw_regression_task(split=split, judgement=judgement)
        imgs = larray.lmap(ImgLoader(ndim=2, shape=(400,400), dtype=dtype, mode='L'),
                           img_paths)
        return imgs, labels



class Facelike(BaseFacelike):
    URL = 'http://dicarlo-faces.s3.amazonaws.com/Facelike_images.zip'
    SHA1 = 'b0cca14daa1b5545bcbf7b0eb40f588fdb25e8ca'
    SUBDIR = 'Facelike_images'
    SUBSETS = [('eye', range(603, 612)),
               ('nose', range(643, 652)),
               ('eye-eye', range(612,620) + range(636, 643)),
               ('eye-nose', range(620, 629) + range(652, 659)),
               ('eye-mouth', range(628, 636) + range(668,675))
              ] 

