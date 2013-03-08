import os
from thoreano.slm import SLMFunction
from skdata import larray
import numpy as np
import tables as tbl

def slm_memmap(desc, X, name, basedir=None, test=None):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    if basedir is None:
        basedir = os.getcwd()
    print('BIPP', X.shape)
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    rval = larray.cache_memmap(feat, name, basedir=basedir)
    return rval

def slm_h5(desc, X, name, basedir=None, save=True):

    if basedir is None:
        basedir = os.getcwd()
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    feat = np.array(feat)
    if save:
        h5file = tbl.openFile(basedir + name + '.h5', mode = "a", title = "model data")
        h5file.createArray(h5file.root, 'features', feat)
        h5file.close()
    return feat
