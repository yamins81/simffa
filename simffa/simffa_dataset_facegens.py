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
            subject_id = np.array([self.meta[i]['subject_id'] for i in len(self.meta)])
            
            self._splits = self.get_splits_by_subjectID(seed, num_subjects, subject_id, num_splits)
        return self._splits

    def get_splits_by_subjectID(seed, num_subjects, subject_id, num_splits):
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
                iMlbl_i = [j for j in range(nIm) if subject_id == unique_id[i]]
                perm_i = rng.permutation(len(iMlbl_i))
                perm = [iMlbl_i[p_i] for p_i in perm_i]
                for ind in perm[:len(perm)/2]:
                    splits['train_' + str(split_id)].append(ind)
                for ind in perm[len(perm)/2 :]:
                    splits['test_' + str(split_id)].append(ind)
        return splits

    def get_label(self, ind):
        label = [self.meta[ind]['subject_id'], self.meta[ind]['express_id'], 
            self.meta[ind]['s'], self.meta[ind]['ty'],self.meta[ind]['tz'],
            self.meta[ind]['rxy'], self.meta[ind]['rxz'],self.meta[ind]['ryz']]
        return label

    def get_images(self):
        tmp = np.array(self.meta)
        inds = range(tmp.shape[0])
        image_paths = [self.meta[ind]['filename'] for ind in inds]
        imgs = larray.lmap(ImgLoader(ndim=2, dtype='uint8', mode='L'), image_paths)
        labels = np.asarray([ (self.meta[ind]['subject_id'], self.meta[ind]['express_id'], 
            self.meta[ind]['s'], self.meta[ind]['ty'],self.meta[ind]['tz'],
            self.meta[ind]['rxy'], self.meta[ind]['rxz'],self.meta[ind]['ryz']) for ind in inds])
    
        IMGS = np.array(imgs)
        LABELS = np.array(labels)
   
        return IMGS, LABELS

    # def save_features(self, config, path_suffix='/hyperopt_features/facegen_fsi', attachments):
        
    #     from simffa.simffa_utils get_features
    #     import tables as tbl

    #     home = self.home()
    #     imgs, labels = self.get_images()
    #     features = get_features(imgs, config)
    #     fs = features.shape
    #     if np.array(fs).shape[0] == 4:
    #         features = features.reshape(fs[0], fs[1]*fs[2]*fs[3])  

    #     np.random.seed()
    #     feature_dir = os.path.join(home, path_suffix)
    #     filename = feature_dir + str(np.random.randint(0,1000000000)) +  '.h5'

    #     h5file = tbl.openFile(filename, mode = 'a')
    #     h5file.createGroup(h5file.root, 'result', title='result')
    #     h5file.createArray(h5file.root.result, 'features', features)
    #     h5file.createArray(h5file.root.result, 'fs', fs)
        
    #     for m in attachments.viewkeys():
    #         h5file.createArray(h5file.root.result, m, attachments[m])

    #     h5file.close()

    #     return filename


class FaceGen_small(FaceGenData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_fg_s.zip'
    SHA1 = 'b97d81032b9ebc07e2612730be2ba701a19aa307'
    SUBDIR = 'DAT_fg_s'

class FaceGen_small_var0(FaceGenData):
    URL = 'http://dicarlocox-datasets.s3.amazonaws.com/DAT_fg_s_var0.zip'
    SHA1 = 'ce2dcddaf9c326e1b4514d385515600476d3a065'
    SUBDIR = 'DAT_fg_s_var0'

