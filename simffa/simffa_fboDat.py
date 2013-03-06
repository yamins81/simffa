# -*- coding: utf-8 -*-
"""A very simple Face Body Object dataset.

   Contains 60 grayscale images (20 each of monkey faces, monkey bodies, 
   and various objects) on pink noise backgrounds.
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

from skdata import larray

from skdata.data_home import get_data_home
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader


class BaseFaceBodyObject(object):

    def __init__(self, meta=None, seed=0, ntrain=10, ntest=10, num_splits=5):

        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits
        self.names = ['Face','Body','Object']

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
            download(url, archive_filename, sha1=sha1)

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

        img_filenames = sorted(os.listdir(self.home(self.SUBDIR)))
        img_filenames = [os.path.join(self.home(self.SUBDIR),x) for x in img_filenames]

        meta = []
        for img_filename in img_filenames:
            img_data = open(img_filename, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            ind = int(os.path.split(img_filename)[1].split('.')[0][2:])
            if ind < 21:
                name = 'Face'
            elif ind < 41:
                name = 'Body'
            else:
                name = 'Object'

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
        labels = int_labels(names)
        return image_paths, labels

    def img_classification_task(self, dtype='uint8', split=None):
        img_paths, labels = self.raw_classification_task(split=split)
        imgs = larray.lmap(ImgLoader(ndim=2, shape=(400,400), dtype=dtype, mode='L'),
                           img_paths)
        return imgs, labels



class FaceBodyObject20110803(BaseFaceBodyObject):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FaceBodyObject_2011_08_03.tar.gz'
    SHA1 = '088387e08ac008a0b8326e7dec1f0a667c8b71d0'
    SUBDIR = 'FaceBodyObject_2011_08_03'



