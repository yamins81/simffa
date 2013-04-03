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

  
class MTData(object):

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
        # all image filenames
        f = open(path.join(home, self.IMG_fn), 'r')
        img_filenames = [[x for x in line.split()] for line in f]
        f.close()
        img_filenames = np.array(img_filenames)
        
        # all psychophysical labels - face
        f = open(path.join(home, self.LABEL1_fn), 'r')
        label_data1  = [[np.double(x) for x in line.split()] for line in f]
        f.close()

        # all psychophysical labels - eye
        f = open(path.join(home, self.LABEL2_fn), 'r')
        label_data2  = [[np.double(x) for x in line.split()] for line in f]
        f.close()

        # all psychophysical labels - nose
        f = open(path.join(home, self.LABEL3_fn), 'r')
        label_data3  = [[np.double(x) for x in line.split()] for line in f]
        f.close()

        # indices of images of interest 
        f = open(path.join(home, self.IMG_OI_fn), 'r')
        img_oi  = [[int(x) for x in line.split()] for line in f]
        f.close()
        img_oi = np.double(img_oi)

        meta = []
        metanames = ['filename', 'id', 'faceLabel', 'eyeLabel', 'noseLabel', 'hash']
        s_im = img_oi.shape
        # maxImgs = 1000
        maxImgs = 50
        nIm = min(maxImgs, s_im[0])
        for i in range(nIm):
            ind = np.int(img_oi[i])
            img_filename = img_filenames[ind-1][0]
            label1 = label_data1[ind-1][1]
            label2 = label_data2[ind-1][1]
            label3 = label_data3[ind-1][1]
            img_fn = path.join(home, img_filename)
            img_data = open(img_fn, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()

            data = dict(filename=img_fn,
                        id=ind,
                        label1=label1,
                        label2=label2,
                        label3=label3,
                        sha1=sha1)
            meta += [data]

            # add translated images ("invariant" dataset)
            for i in range(2):
                new_tag = '_i'+str(i+1)+'.png'
                invar_fn = img_fn.replace('.png', new_tag)
                img_data = open(invar_fn, 'rb').read()
                sha1 = hashlib.sha1(img_data).hexdigest()
                data2 = dict(filename=invar_fn,
                            id=ind,
                            label1=label1,
                            label2=label2,
                            label3=label3,
                            sha1=sha1)
                meta += [data2]

        # meta = tb.tabarray(records=meta, names=metanames)
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

    # def generate_splits(self, seed, ntrain, ntest, num_splits, labelset=None, catfunc=None):
    #     meta = self.meta
    #     if labelset is not None:
    #         assert catfunc is not None
    #     else:
    #         labelset = self.names
    #         catfunc = lambda x : x['name']

    #     ntrain = self.ntrain
    #     ntest = self.ntest
    #     rng = np.random.RandomState(seed)
    #     splits = {}
    #     for split_id in range(num_splits):
    #         splits['train_' + str(split_id)] = []
    #         splits['test_' + str(split_id)] = []
    #         for label in labelset:
    #             cat = [m for m in meta if catfunc(m) == label]
    #             L = len(cat)

    #             assert L >= ntrain + ntest, 'category %s too small' % name
    #             perm = rng.permutation(L)
    #             for ind in perm[:ntrain]:
    #                 splits['train_' + str(split_id)].append(cat[ind]['filename'])
    #             for ind in perm[ntrain: ntrain + ntest]:
    #                 splits['test_' + str(split_id)].append(cat[ind]['filename'])
    #     return splits


    # def generate_regression_splits(self, labels, seed, ntrain, ntest, num_splits):
    #     meta = self.meta
    #     nIm = np.array(meta).shape[0]
    #     rng = np.random.RandomState(seed)
    #     splits = {}
    #     for split_id in range(num_splits):
    #         splits['train_' + str(split_id)] = []
    #         splits['test_' + str(split_id)] = []
            
    #         perm = rng.permutation(nIm)
    #         for ind in perm[:ntrain]:
    #             splits['train_' + str(split_id)].append(ind)
    #         for ind in perm[ntrain: ntrain + ntest]:
    #             splits['test_' + str(split_id)].append(ind)

    #     return splits

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())


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


class MTData_Feb222013(MTData):
    URL = './'
    SHA1 = '088387e08ac008a0b8326e7dec1f0a667c8b71d0'
    SUBDIR = 'DAT_mt'
    IMG_fn = 'img_all.txt'
    LABEL1_fn = 'psyFaceMag_20121012_210.txt'
    LABEL2_fn = 'psyEyeMag_20121228_285.txt'
    LABEL3_fn = 'psyNoseMag_20130201_285.txt'
    # IMG_OI_fn = 'img_oi818.txt'
    IMG_OI_fn = 'img_oi408.txt'

class MTData_March082013(MTData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/simffa_dat.zip'
    SHA1 = '1cb9e893d7a582040aef898d00f3d370bf274efe'
    SUBDIR = 'DAT_mt'
    IMG_fn = 'img_all.txt'
    LABEL1_fn = 'psyFaceMag_20121012_210.txt'
    LABEL2_fn = 'psyEyeMag_20121228_285.txt'
    LABEL3_fn = 'psyNoseMag_20130201_285.txt'
    # IMG_OI_fn = 'img_oi818.txt'
    IMG_OI_fn = 'img_oi408.txt'




# def get_transformed_image(img):
#     cimg = Image.fromarray(img).copy()
#     if np.random.random() > 0.5:
#         cimg = cimg.transpose(Image.FLIP_LEFT_RIGHT)
#     if np.random.random() > 0.5:
#         cimg = cimg.transpose(Image.FLIP_TOP_BOTTOM) 
#     cimg = cimg.rotate(360*np.random.random())

#     xoffset = np.int(np.random.random()*200);
#     yoffset = np.int(np.random.random()*200);
#     cimg = cimg.offset(xoffset, yoffset)
    
#     return cimg

