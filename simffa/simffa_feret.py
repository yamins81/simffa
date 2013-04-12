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
import skdata as skd
from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader

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

        meta = []
        for img_filename in img_filenames:
            img_data = open(img_filename, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            subject_id = os.path.split(img_filename)[1].split('.')[0][:5]
            pose_id = os.path.split(img_filename)[1].split('.')[0][5:7]
            
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

        # imgs = larray.lmap(ImgLoader(ndim=2, shape=(400,400), dtype='uint8', mode='L'),
                           # image_paths)
        labels = np.asarray([ (self.meta[ind]['subject_id'], self.meta[ind]['pose_id']) for ind in inds])
    
        IMGS = np.array(imgs)
        LABELS = np.array(labels)
   
        return IMGS, LABELS


class FERET(FeretData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FERET.zip'
    SHA1 = '3666828effe1cc77b6297406f1e0f9799137068e'
    SUBDIR = 'FERET'


