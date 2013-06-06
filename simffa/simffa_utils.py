import os
from thoreano.slm import SLMFunction
from skdata import larray
from skdata.data_home import get_data_home
import numpy as np
import tables as tbl


def get_features(X, config, verbose=False, dirname=None, fname_tag=None):
    features = slm_h5(
                    desc=config['desc'],
                    X=X,
                    basedir=dirname,
                    name=fname_tag, 
                    save=False) 
    features = np.array(features)
    return features

def save_features(path_suffix, attachments):
        home = get_data_home()
        np.random.seed()
        feature_dir = os.path.join(home, path_suffix)
        filename = feature_dir + str(np.random.randint(0,1000000000)) +  '.h5'

        h5file = tbl.openFile(filename, mode = 'a')
        h5file.createGroup(h5file.root, 'result', title='result')
        for m in attachments.viewkeys():
            h5file.createArray(h5file.root.result, m, attachments[m])
        h5file.close()

        return filename

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

def slm_h5(desc, X, name, basedir=None, save=False):

    if basedir is None:
        basedir = os.getcwd()
    
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    
    if save:
        h5file = tbl.openFile(basedir + name + '.h5', mode = "a", title = "model data")
        h5file.createArray(h5file.root, 'features', feat)
        h5file.close()
    return feat
