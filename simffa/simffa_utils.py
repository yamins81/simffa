import os
from thoreano.slm import SLMFunction
from skdata import larray

def slm_memmap(desc, X, name, basedir=None, test=None):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    if basedir is None:
        basedir = os.getcwd()
    print('BIPP', X.shape)
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X, verbose=False)
    rval = larray.cache_memmap(feat, name, basedir=basedir, test=test)
    return rval