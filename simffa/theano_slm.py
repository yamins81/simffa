import sys
import time
import os
import copy
import itertools
import tempfile
import os.path as path
import hashlib
import cPickle

import Image
import numpy as np
from bson import BSON, SON
import asgd  # use master branch from https://github.com/jaberg/asgd
from thoreano.slm import TheanoSLM
from pythor3.model import SequentialLayeredModel

from early_stopping import fit_w_early_stopping, EarlyStopping


TEST = False
TEST_NUM = 200
DEFAULT_TLIMIT = 35


def son_to_py(son):
    """ turn son keys (unicode) into str
    """
    if isinstance(son, SON):
        return dict([(str(k), son_to_py(v)) for k, v in son.items()])
    elif isinstance(son, list):
        return [son_to_py(s) for s in son]
    elif isinstance(son, basestring):
        return str(son)
    else:
        return son


def use_memmap(size):
    if size < 3e8:
        memmap = False
    else:
        memmap = True
    return memmap


class ExtractedFeatures(object):
    def __init__(self, X, feature_shps, batchsize, slms, filenames,
                 tlimit=DEFAULT_TLIMIT, file_out = False):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """

        self.filenames = []
        self.features = []
        self.feature_shps = feature_shps

        for feature_shp, filename, slm in zip(feature_shps, filenames, slms):
            size = 4 * np.prod(feature_shp)
            print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
            memmap = file_out or use_memmap(size)
            if memmap:
                print('Creating memmap %s for features of shape %s' % (
                                                      filename, str(feature_shp)))
                features_fp = np.memmap(filename,
                    dtype='float32',
                    mode='w+',
                    shape=feature_shp)
            else:
                print('Using memory for features of shape %s' % str(feature_shp))
                features_fp = np.empty(feature_shp,dtype='float32')

            if TEST:
                print('TESTING')

            i = 0
            t0 = time.time()
            while not TEST or i < 10:
                if i + batchsize >= len(X):
                    assert i < len(X)
                    xi = np.asarray(X[-batchsize:])
                    done = True
                else:
                    xi = np.asarray(X[i:i+batchsize])
                    done = False
                t1 = time.time()
                feature_batch = slm.process_batch(xi)
                if TEST:
                    print('compute: ', time.time() - t1)
                t2 = time.time()
                delta = max(0, i + batchsize - len(X))
                assert np.all(np.isfinite(feature_batch))
                features_fp[i:i+batchsize-delta] = feature_batch[delta:]
                if TEST:
                    print('write: ', time.time() - t2)
                if done:
                    break

                i += batchsize
                if (i // batchsize) % 50 == 0:
                    t_cur = time.time() - t0
                    t_per_image = (time.time() - t0) / i
                    t_tot = t_per_image * X.shape[0]
                    if tlimit is not None and t_tot / 60.0 > tlimit:
                        raise TooLongException(t_tot/60.0, tlimit)
                    print 'get_features_fp: %i / %i  mins: %.2f / %.2f ' % (
                            i , len(X),
                            t_cur / 60.0, t_tot / 60.0)
            # -- docs hers:
            #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
            #    say that deletion is the way to flush changes !?
            if memmap:
                del features_fp
                self.filenames.append(filename)
                features_fp = np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=feature_shp)
                self.features.append(features_fp)
            else:
                self.filenames.append('')
                self.features.append(features_fp)

    def __enter__(self):
        return self.features

    def __exit__(self, *args):
        for filename in self.filenames:
            if filename:
                os.remove(filename)


class TheanoExtractedFeatures(ExtractedFeatures):
    def __init__(self, X, batchsize, configs, filenames, tlimit=DEFAULT_TLIMIT,
                 use_theano=True):
        slms = [slm_from_config(config, X.shape, batchsize) for config in configs]
        feature_shps = [(X.shape[0],) + slm.pythor_out_shape for slm in slms]
        super(TheanoExtractedFeatures, self).__init__(X, feature_shps,
                batchsize,
                slms,
                filenames,
                tlimit=tlimit)


class FeatureExtractor(object):
    def __init__(self, X, slm,
            tlimit=float('inf'),
            batchsize=4,
            filename='FeatureExtractor.npy',
            TEST=False):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """
        self.filename = filename
        self.batchsize = batchsize
        self.tlimit = tlimit
        self.X = X
        self.slm = slm
        self.verbose = False
        self.n_to_extract = len(X)
        if TEST:
            print('FeatureExtractor running in TESTING mode')
            self.verbose = True
            self.n_to_extract = 10 * batchsize
        assert self.n_to_extract <= len(X)

        # -- convenience
        self.feature_shp = (self.n_to_extract,) + self.slm.pythor_out_shape

    def __enter__(self):
        if self.filename:
            self.features = self.compute_features(use_memmap=True)
        else:
            self.features = self.compute_features(use_memmap=False)
        return self.features

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)
        del self.features

    def extract_to_memmap(self):
        """
        Allocate a memmap, fill it with extracted features, return r/o view.
        """
        filename = self.filename
        feature_shp = self.feature_shp
        print('Creating memmap %s for features of shape %s' % (
                                              filename,
                                              str(feature_shp)))
        features_fp = np.memmap(filename,
            dtype='float32',
            mode='w+',
            shape=feature_shp)
        info = open(filename+'.info', 'w')
        cPickle.dump(('float32', feature_shp), info)
        del info

        self.extract_to_storage(features_fp)

        # -- docs here:
        #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        #    say that deletion is the way to flush changes !?
        del features_fp
        rval = np.memmap(self.filename,
            dtype='float32',
            mode='r',
            shape=feature_shp)
        return rval

    def extract_to_storage(self, arr):
        """
        Fill arr with the first len(arr) features of self.X.
        """
        assert len(arr) <= len(self.X)
        batchsize = self.batchsize
        tlimit = self.tlimit
        print('Total size: %i bytes (%.2f GB)' % (
            arr.size * arr.dtype.itemsize,
            arr.size * arr.dtype.itemsize / float(1e9)))
        i = 0
        t0 = time.time()
        while True:
            if i + batchsize >= len(arr):
                assert i < len(arr)
                xi = np.asarray(self.X[-batchsize:])
                done = True
            else:
                xi = np.asarray(self.X[i:i+batchsize])
                done = False
            t1 = time.time()
            feature_batch = self.slm.process_batch(xi)
            if self.verbose:
                print('compute: ', time.time() - t1)
            t2 = time.time()
            delta = max(0, i + batchsize - len(arr))
            assert np.all(np.isfinite(feature_batch))
            arr[i:i + batchsize - delta] = feature_batch[delta:]
            if self.verbose:
                print('write: ', time.time() - t2)
            if done:
                break

            i += batchsize
            if (i // batchsize) % 50 == 0:
                t_cur = time.time() - t0
                t_per_image = (time.time() - t0) / i
                t_tot = t_per_image * len(arr)
                if tlimit is not None and t_tot / 60.0 > tlimit:
                    raise TooLongException(t_tot/60.0, tlimit)
                print 'extraction: %i / %i  mins: %.2f / %.2f ' % (
                        i , len(arr),
                        t_cur / 60.0, t_tot / 60.0)

    def compute_features(self, use_memmap=None):
        if use_memmap is None:
            size = np.prod(self.feature_shp) * 4
            use_memmap = (size > 3e8)  # 300MB cutoff

        if use_memmap:
            return self.extract_to_memmap()
        else:
            print('Using memory for features of shape %s' % str(self.feature_shp))
            arr = np.empty(self.feature_shp, dtype='float32')
            self.extract_to_storage(arr)
            return arr


def slm_from_config(config, X_shape, batchsize, use_theano=True):
    config = son_to_py(config)
    desc = copy.deepcopy(config['desc'])
    interpret_model(desc)

    if use_theano:
        if len(X_shape) == 3:
            t_slm = TheanoSLM(
                    in_shape=(batchsize,) + X_shape[1:] + (1,),
                    description=desc)
        elif len(X_shape) == 4:
            t_slm = TheanoSLM(
                    in_shape=(batchsize,) + X_shape[1:],
                    description=desc)
        else:
            raise NotImplementedError()
        slm = t_slm
        # -- pre-compile the function to not mess up timing later
        slm.get_theano_fn()
    else:
        cthor_sse = {'plugin':'cthor', 'plugin_kwargs':{'variant':'sse'}}
        cthor = {'plugin':'cthor', 'plugin_kwargs':{}}
        slm = SequentialLayeredModel(X_shape[1:], desc,
                                     plugin='passthrough',
                                     plugin_kwargs={'plugin_mapping': {
                                         'fbcorr': cthor,
                                          'lnorm' : cthor,
                                          'lpool' : cthor,
                                     }})

    return slm


class TooLongException(Exception):
    """model takes too long to evaluate"""
    def msg(tot, cutoff):
        return 'Would take too long to execute model (%f mins, but cutoff is %s mins)' % (tot, cutoff)


def flatten(x):
    return list(itertools.chain(*x))

def interpret_activ(filter, activ):
    n_filters = filter['initialize']['n_filters']
    generator = activ['generate'][0].split(':')
    vals = activ['generate'][1]

    if generator[0] == 'random':
        dist = generator[1]
        if dist == 'uniform':
            mean = vals['mean']
            delta = vals['delta']
            seed = vals['rseed']
            rng = np.random.RandomState(seed)
            low = mean-(delta/2)
            high = mean+(delta/2)
            size = (n_filters,)
            activ_vec = rng.uniform(low=low, high=high, size=size)
        elif dist == 'normal':
            mean = vals['mean']
            stdev = vals['stdev']
            seed = vals['rseed']
            rng = np.random.RandomState(seed)
            size = (n_filters,)
            activ_vec = rng.normal(loc=mean, scale=stdev, size=size)
        else:
            raise ValueError, 'distribution not recognized'
    elif generator[0] == 'fixedvalues':
        values = vals['values']
        num = n_filters / len(values)
        delta = n_filters - len(values)*num
        activ_vec = flatten([[v]*num for v in values] + [[values[-1]]*delta])
    else:
        raise ValueError, 'not recognized'

    return activ_vec

def interpret_model(desc):
    for layer in desc:
        for (ind,(opname,opparams)) in enumerate(layer):

            if opname == 'fbcorr_h':
                kw = opparams['kwargs']
                if hasattr(kw.get('min_out'),'keys'):
                    kw['min_out'] = interpret_activ(opparams, kw['min_out'])
                if hasattr(kw.get('max_out'),'keys'):
                    kw['max_out'] = interpret_activ(opparams, kw['max_out'])
            elif opname == 'lpool_h':
                kw = opparams['kwargs']
                if hasattr(kw.get('order'),'keys'):
                    kw['order'] = interpret_activ(layer[0][1], kw['order'])

            if opname in ['fbcorr', 'fbcorr_h']:
                init = opparams['initialize']
                if init.has_key('filter_size'):
                    sz = init.pop('filter_size')
                    init['filter_shape'] = (2*sz+1, 2*sz+1)
            elif opname in ['lnorm', 'lnorm_h']:
                init = opparams['kwargs']
                if init.has_key('inker_size'):
                    sz = init.pop('inker_size')
                    init['inker_shape'] = (2*sz+1, 2*sz+1)
                if init.has_key('outker_size'):
                    sz = init.pop('outker_size')
                    init['outker_shape'] = (2*sz+1, 2*sz+1)
            elif opname in ['lpool', 'lpool_h']:
                init = opparams['kwargs']
                if init.has_key('ker_size'):
                    sz = init.pop('ker_size')
                    init['ker_shape'] = (2*sz+1, 2*sz+1)
