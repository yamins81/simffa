import os
import shutil
import hashlib
import cPickle

import numpy as np
from skdata import larray
from skdata.utils import download, extract, int_labels
from skdata.utils.image import ImgLoader
from skdata.data_home import get_data_home

class FaceGenData(object):

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

    @property
    def filenames(self):
        return [m['filename'] for m in self.meta]

    def _get_meta(self):
        home = self.home()

        meta_fn = os.path.join(self.home(self.SUBDIR),'metadata.pkl') 
        file = open(meta_fn,'r')
        metadata = cPickle.load(file)
        meta = []
        for mi in range(len(metadata)):
            img_id = metadata['id'][mi]

            subject_id = int(img_id[4:7])
            # if subject_id < 4:
            #     continue

            express_id = int(img_id[8:-1])

            img_fn = os.path.join(self.home(self.SUBDIR),img_id + '.png') 
            img_data = open(img_fn, 'rb').read()
            sha1 = hashlib.sha1(img_data).hexdigest()
            
            ty = metadata['ty'][mi][0]
            tz = metadata['tz'][mi][0]
            s = metadata['s'][mi][0]
            rxy = metadata['rxy'][mi][0]
            rxz = metadata['rxz'][mi][0]
            ryz = metadata['ryz'][mi][0]

            data = dict(filename=img_fn,
                        subject_id=subject_id,
                        express_id=express_id,
                        ty=ty,tz=tz,s=s,rxy=rxy,rxz=rxz,ryz=ryz,
                        sha1=sha1)
            meta += [data]

        return meta

    @property
    def splits(self):
        if not hasattr(self, '_splits'):
            np.random.seed()
            seed = np.random.randint(1000000) 

            num_subjects = 8
            num_splits = 10
            # subject_id = np.array([self.meta[i]['subject_id'] for i in range(len(self.meta))])
            
            self._splits = self.get_splits_by_subjectID(seed, num_subjects,num_splits)
        return self._splits

    def get_splits_by_subjectID(self, seed, num_subjects, num_splits):

        subject_id = np.array([self.meta[i]['subject_id'] for i in range(len(self.meta))])
        unique_id = np.unique(subject_id)
        if num_subjects > unique_id.shape[0]:
            num_subjects = unique_id.shape[0]

        nIm = np.array(subject_id).shape[0]
        rng = np.random.RandomState(seed)
        splits = {}

        # split half-half based on subject id, no control over expression and pose
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

    # def get_label(self, ind):
    #     label = [self.meta[ind]['subject_id'], self.meta[ind]['express_id'], 
    #         self.meta[ind]['s'], self.meta[ind]['ty'],self.meta[ind]['tz'],
    #         self.meta[ind]['rxy'], self.meta[ind]['rxz'],self.meta[ind]['ryz']]
    #     return label

    def get_labels(self):
        labels = {}
        labels['subject'] = [m['subject_id'] for m in self.meta]
        labels['express'] = [m['express_id'] for m in self.meta]
        return labels

    def get_images(self, preproc=None):
        self.fetch()
        if preproc == None:
            dtype = 'float32'
            mode = 'L'
            size = (256,256)
            normalize = True
        else:
            dtype = preproc['dtype']
            mode = preproc['mode']
            size = tuple(preproc['size'])
            normalize = preproc['normalize']
        return larray.lmap(ImgLoaderResizer(inshape=(256, 256),
                                            shape=size,
                                            dtype=dtype,
                                            normalize=normalize,
                                            mode=mode),
                                self.filenames)

class FaceGen_small(FaceGenData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_fg_s.zip'
    SHA1 = 'b97d81032b9ebc07e2612730be2ba701a19aa307'
    SUBDIR = 'DAT_fg_s'

class FaceGen_small_var0(FaceGenData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_fg_s_var0.zip'
    SHA1 = 'ce2dcddaf9c326e1b4514d385515600476d3a065'
    SUBDIR = 'DAT_fg_s_var0'


class ImgLoaderResizer(object):
    """
    """
    def __init__(self,
                 inshape,
                 shape=None,
                 ndim=None,
                 dtype='float32',
                 normalize=True,
                 mode='L'
                 ):
        self.inshape = inshape
        shape = tuple(shape)
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.normalize = normalize
        self.mode = mode

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
        if im.mode != self.mode:
            im = im.convert(self.mode)
        assert im.size == self.inshape[:2]
        if max(im.size) != self._shape[0]:
            m = self._shape[0]/float(max(im.size))
            new_shape = (int(round(im.size[0]*m)), int(round(im.size[1]*m)))
            im = im.resize(new_shape, Image.ANTIALIAS)
        rval = np.asarray(im, self._dtype)
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            if 'float' in str(self._dtype):
                rval /= 255.0
        assert rval.shape == self._shape
        return rval

