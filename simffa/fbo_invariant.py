# -*- coding: utf-8 -*-
"""Invariant Face Body Object dataset.

   Invariance-modified version of 60 grayscale images (20 each of monkey faces, 
   monkey bodies, and various objects) on pink noise backgrounds.
"""

# Copyright (C) 2012
# Authors:  Dan Yamins

# License: Simplified BSD


import os
from os import path
import shutil
from glob import glob
import hashlib
import cPickle

import numpy as np

import skdata.larray as larray
import Image
from skdata.data_home import get_data_home
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader
import hyperopt.gdist as gd


def get_modified_image(cimg, bimg, inv_data):
    cimg = cimg.copy()
    bimg = bimg.copy()
    if inv_data['flip_lr']:
        cimg = cimg.transpose(Image.FLIP_LEFT_RIGHT)
    if inv_data['flip_ud']:
        cimg = cimg.transpose(Image.FLIP_TOP_BOTTOM) 
    cimg = cimg.rotate(360*inv_data['rot'])
    scale = inv_data['scale']
    origsize = cimg.size
    newsize = (int(round(cimg.size[0]*scale)), int(round(cimg.size[1]*scale)))
    cimg = cimg.resize(newsize)
    
    bsize = bimg.size
    xctr, yctr = (bsize[0]/2, bsize[1]/2)
    xpos = xctr + int(round(inv_data['xpos']*origsize[0]/2))
    ypos = yctr + int(round(inv_data['ypos']*origsize[1]/2))
    box = (xpos - newsize[0]/2, 
           ypos - newsize[1]/2, 
           xpos - newsize[0]/2 + newsize[0],
           ypos - newsize[1]/2 + newsize[1])
    
    mask = cimg.convert('RGBA').split()[-1]
    print(box)
    bimg.paste(cimg, box, mask)

    return bimg
        
    
    
class BaseFaceBodyObjectInvariant(object):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/FaceBodyObjectInvariant_2011_08_03.tar.gz'
    NUM_GENERATE_PER_ORIGINAL = 3
    SHA1 = 'c688b13f1f9e6723c5a99b0ccf477cf2805b236f'
    SUBDIR = 'FaceBodyObjectInvariant_2011_08_03'
    IMDIR = 'Generated_Images'

    def __init__(self, meta=None, seed=0, ntrain=10, ntest=10,
                 num_splits=5, gseed=0, make_original=False):

        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits
        self.names = ['Face','Body','Object']
        self.make_original = make_original
        
        self.genson_template = gd.gDist(self.genson_string)
        self.genson_template.seed(gseed)

        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__


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
        
        imdir = self.home(self.IMDIR)
        if not path.exists(imdir):
            os.mkdir(imdir)
        
            #generate images and metadata pkl
            background_dir = os.path.join(self.home(self.SUBDIR),'pink_noise')
            cutout_dir =  os.path.join(self.home(self.SUBDIR),'cutouts')
            cutout_files = filter(lambda x: x.startswith('im'), os.listdir(cutout_dir))
            metadata = []
            for cf in cutout_files:
                print os.path.join(cutout_dir,cf)
                ind = int(os.path.split(cf)[1].split('.')[0][2:])
                bg = 'impink' +  str(ind) + '.png'
                cimg = Image.open(os.path.join(cutout_dir,cf))
                bimg = Image.open(os.path.join(background_dir,bg))
                mask = cimg.convert('RGBA').split()[-1]
                
                #original image
                
                name = 'Face' if ind < 21 else 'Body' if ind < 41 else 'Object'
                if self.make_original:
                    im = bimg.copy()
                    im.paste(cimg.copy(), None, mask)
                    
                    filename = os.path.join(imdir,'original_' + str(ind)) + '.png'
                    im.save(filename)
                    sha1 = hashlib.sha1(open(filename,'rb').read()).hexdigest()
                    meta = dict(name=name,
                                sha1=sha1,
                                original=1,
                                id='original_' + str(ind),
                                filename=filename)
                    metadata.append(meta)
                 
                for _ind in range(self.NUM_GENERATE_PER_ORIGINAL):
                    inv_data = self.genson_template.sample()
                    im = get_modified_image(cimg, bimg, inv_data)
                    filename = os.path.join(imdir,'generated_' + str(ind) + '_' + str(_ind)) + '.png'
                    im.save(filename)
                    sha1 = hashlib.sha1(open(filename,'rb').read()).hexdigest()                
                    meta = dict(name=name,
                            sha1=sha1,
                            original=0,
                            id='generated_' + str(ind) + '_' + str(_ind),
                            filename=filename)
                    meta.update(inv_data)
                    metadata.append(meta)
                    
            metapath = self.home('meta.pkl')
            mf = open(metapath,'w')
            cPickle.dump(metadata, mf)
            mf.close()
        
        
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
        metapath = self.home('meta.pkl')
        meta = cPickle.load(open(metapath))
        filenames = np.array([m['filename'] for m in meta])
        fsort = filenames.argsort()
        meta = [meta[s] for s in fsort]
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


class FaceBodyObject20110803Invariant0(BaseFaceBodyObjectInvariant):
    genson_string = """{"scale": uniform(0.9, 1),
                        "xpos": uniform(-0.1, 0.1),
                        "ypos": uniform(-0.1, 0.1),
                        "rot": uniform(-0.1, 0.1),
                        "flip_lr": 0,
                        "flip_ud": 0}"""


class FaceBodyObject20110803Invariant1(BaseFaceBodyObjectInvariant):
    genson_string = """{"scale": uniform(0.75, 1),
                        "xpos": uniform(-0.25, 0.25),
                        "ypos": uniform(-0.25, 0.25),
                        "rot": uniform(-0.25, 0.25),
                        "flip_lr": choice([0, 1]),
                        "flip_ud": 0}"""


class FaceBodyObject20110803Invariant2(BaseFaceBodyObjectInvariant):
    genson_string = """{"scale": uniform(0.5, 1),
                        "xpos": uniform(-0.5, 0.5),
                        "ypos": uniform(-0.5, 0.5),
                        "rot": uniform(-0.5, 0.5),
                        "flip_lr": choice([0, 1]),
                        "flip_ud": choice([0, 1])}"""
                        

class FaceBodyObject20110803InvariantFlip(BaseFaceBodyObjectInvariant):
    genson_string = """{"scale": 1,
                        "xpos": 0,
                        "ypos": 0,
                        "rot": 0,
                        "flip_lr": choice([0, 1]),
                        "flip_ud": choice([0, 1])}"""
    
