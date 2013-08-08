import os
import Image
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


class ImgLoaderResizer(object):
    """
    """
    def __init__(self,
                 inshape,
                 shape=None,
                 ndim=None,
                 dtype='float32',
                 normalize=True,
                 crop=None,
                 mask=None):
        self.inshape = inshape
        assert len(shape) == 2
        shape = tuple(shape)
        if crop is None:
            crop = (0, 0, self.inshape[0], self.inshape[1])
        assert len(crop) == 4
        crop = tuple(crop)
        l, t, r, b = crop
        assert 0 <= l < r <= self.inshape[0]
        assert 0 <= t < b <= self.inshape[1]
        self._crop = crop
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.normalize = normalize
        self.mask = mask

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        if im.mode != 'L':
            im = im.convert('L')
        assert im.size == self.inshape
        if self.mask is not None:
            mask = self.mask
            tmask = ImageOps.invert(mask.convert('RGBA').split()[-1])
            im = Image.composite(im, mask, tmask).convert('L')
        if self._crop != (0, 0,) + self.inshape:
            im = im.crop(self._crop)
        l, t, r, b = self._crop
        assert im.size == (r - l, b - t)
        if max(im.size) != self._shape[0]:
            m = self._shape[0]/float(max(im.size))
            new_shape = (int(round(im.size[0]*m)), int(round(im.size[1]*m)))
            im = im.resize(new_shape, Image.ANTIALIAS)
        imval = np.asarray(im, 'float32')
        rval = np.zeros(self._shape)
        ctr = self._shape[0]/2
        cxmin = ctr - imval.shape[0] / 2
        cxmax = ctr - imval.shape[0] / 2 + imval.shape[0]
        cymin = ctr - imval.shape[1] / 2
        cymax = ctr - imval.shape[1] / 2 + imval.shape[1]
        rval[cxmin:cxmax,cymin:cymax] = imval
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape == self._shape
        return rval
