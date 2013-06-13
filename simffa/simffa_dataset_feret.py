# -*- coding: utf-8 -*-
# Rishi Rajalingham (2013)
# License: Simplified BSD

import os
import shutil
import hashlib
import glob

import numpy as np
from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader
from skdata.data_home import get_data_home

class FeretData(object):

    def __init__(self, meta=None):
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

        # img_filenames = sorted(os.listdir(self.home(self.SUBDIR)))
        img_filenames = glob.glob(os.path.join(self.home(self.SUBDIR),'*.png'))
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

    @property
    def splits():
        if not hasattr(self, '_splits'):
            np.random.seed()
            seed = np.random.randint(1000000) 
            num_subjects = 8
            num_splits = 10
            
            
            self._splits = self.get_splits_by_subjectID(seed, num_subjects, num_splits)
        return self._splits

    def get_splits_by_subjectID(self, seed, num_subjects, num_splits):
        subject_id = np.array([self.meta[ind]['subject_id'] for i in range(len(meta))])
        unique_id = np.unique(subject_id)
        if num_subjects > unique_id.shape[0]:
            num_subjects = unique_id.shape[0]

        nIm = np.array(subject_id).shape[0]
        rng = np.random.RandomState(seed)
        splits = {}

        # split half-half based on subject id, no control over pose
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []

            for i in range(num_subjects):
                iMlbl_i = [j for j in range(nIm) if subject_id[j] == unique_id[i]]
                perm_i = rng.permutation(len(iMlbl_i))
                perm = [iMlbl_i[p_i] for p_i in perm_i]
                for ind in perm[:len(perm)/2]:
                    splits['train_' + str(split_id)].append(ind)
                for ind in perm[len(perm)/2 :]:
                    splits['test_' + str(split_id)].append(ind)
        return splits


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
    SUBDIR = './'

class FERET_gray(FeretData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FERET_gray.zip'
    SHA1 = '9c694eb2ae49022566c641a04e5eea776928869a'
    SUBDIR = './'



