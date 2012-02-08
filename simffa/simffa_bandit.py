"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy
import cPickle

import pymongo as pm
import numpy as np
import scipy.stats as sp_stats

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
from thoreano.slm import slm_from_config, FeatureExtractor
from thoreano.classifier import train_scikits
import skdata.fbo
from yamutils.stats import pearsonr, spearmanr

import simffa_params
import simffa_datasets
import fbo_invariant


####################
########Common stuff
####################

def get_features(X, config, verbose=False):
    batchsize = 4
    slm = slm_from_config(config, X.shape, batchsize=batchsize)
    extractor = FeatureExtractor(X, slm, batchsize=batchsize, verbose=verbose)
    features = extractor.compute_features(use_memmap=False)
    return features
    

def traintest(dataset, features, seed=0, ntrain=10, ntest=10, num_splits=5, catfunc=None):
    if catfunc is None:
        catfunc = lambda x : x['name']
    Xr = np.array([m['filename'] for m in dataset.meta])
    labels = np.array([catfunc(m) for m in dataset.meta])
    labelset = set(labels)
    splits = dataset.generate_splits(seed, ntrain, ntest, num_splits, labelset=labelset, catfunc=catfunc)
    results = []
    for ind in range(num_splits):
        train_split = np.array(splits['train_' + str(ind)])
        test_split = np.array(splits['test_' + str(ind)])
        train_inds = np.searchsorted(Xr,train_split)
        test_inds = np.searchsorted(Xr,test_split)
        train_X = features[train_inds]
        test_X = features[test_inds]
        train_y = labels[train_inds]
        test_y = labels[test_inds]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'liblinear', regression=False)
        results.append(result)
    return results


def regression_traintest(dataset, features, regfunc, seed=0, ntrain=80, ntest=30, num_splits=5):
    Xr = np.array([m['filename'] for m in dataset.meta])
    labels = np.array([regfunc(m) for m in dataset.meta])
    splits = dataset.generate_splits(seed, ntrain, ntest, num_splits)
    results = []
    for ind in range(num_splits):
        train_split = np.array(splits['train_' + str(ind)])
        test_split = np.array(splits['test_' + str(ind)])
        train_inds = np.searchsorted(Xr,train_split)
        test_inds = np.searchsorted(Xr,test_split)
        train_X = features[train_inds]
        test_X = features[test_inds]
        train_y = labels[train_inds]
        test_y = labels[test_inds]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        result = train_scikits(train_Xy, test_Xy, 'linear_model.LassoCV', regression=True)
        results.append(result)
    return results
    

def evaluate_just_FSI(dataset, config, train=False, **training_data):
    record, FSI = evaluate_FSI(dataset, config, train=train, **training_data)
    for k in ['F_s_avg', 'BO_s_avg', 'FSI_s_avg']:
        spatial_averages[k] = record.pop(k)       
    return record, FSI
    

    
    
def evaluate_FSI(dataset, config, train=True, **training_data):
    X, y = dataset.img_classification_task()
    features = get_features(X, config, verbose=True)
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]    
    meta = dataset.meta
    face_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] == 'Face']
    bo_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] != 'Face']
    
    record = {}
    record['num_features'] = num_features
    record['feature_shape'] = fs
    thresholds = np.arange(0,1,.01)
    record['thresholds'] = thresholds.tolist()
    
    F = features[face_inds].mean(0)
    BO = features[bo_inds].mean(0)
    record['F_s_avg'] = F.mean(2).tolist()
    record['BO_s_avg'] = BO.mean(2).tolist()
  
    FSI = (F - BO) / (np.abs(F) + np.abs(BO))
    FSI_counts = [len((FSI > thres).nonzero()[0]) for thres in thresholds]
    FSI_fractions = [c/ float(num_features) for c in FSI_counts]
    record['fsi_fractions'] = FSI_fractions
    record['FSI_s_avg'] = FSI.mean(2).tolist()
    record['Face_selective_s_avg'] = (FSI > .333).astype(np.float).mean(2).tolist()

    F_rect = np.abs(features[face_inds]).mean(0)
    BO_rect = np.abs(features[bo_inds]).mean(0)
    FSI_rect = (F_rect - BO_rect) / (F_rect + BO_rect)
    FSI_rect_counts = [len((FSI_rect > thres).nonzero()[0]) for thres in thresholds]
    FSI_rect_fractions = [c/ float(num_features) for c in FSI_rect_counts]
    record['rectified_fsi_fractions'] = FSI_rect_fractions
    record['rectified_FSI_s_avg'] = FSI_rect.mean(2).tolist()
    record['rectified_Face_selective_s_avg'] = (FSI_rect > .333).astype(np.float).mean(2).tolist()

    features_shifted = features - features.min()
    F_shifted = features_shifted[face_inds].mean(0)
    BO_shifted = features_shifted[bo_inds].mean(0)
    FSI_shifted = (F_shifted - BO_shifted) / (F_shifted + BO_shifted)
    FSI_shifted_counts = [len((FSI_shifted > thres).nonzero()[0]) for thres in thresholds]
    FSI_shifted_fractions = [c/ float(num_features) for c in FSI_shifted_counts]
    record['shifted_fsi_fractions'] = FSI_shifted_fractions
    record['shifted_FSI_s_avg'] = FSI_shifted.mean(2).tolist()
    record['shifted_Face_selective_s_avg'] = (FSI_shifted > .333).astype(np.float).mean(2).tolist()
    
    dprime = (F - BO) / features.std(0)
    dprime_h, dprime_b = np.histogram(dprime, bins=50)
    record['dprime_hist'] = dprime_h.tolist()
    record['dprime_bins'] = dprime_b.tolist()
    record['dprime_selective_fraction'] = float(len((dprime.flatten() > 1).nonzero()[0]) / float(num_features))
    record['dprime_selective_s_avg'] = (dprime > 1).astype(np.float).mean(2).tolist()
    
    Z = np.row_stack([features[face_inds], features[bo_inds]])
    R = Z.argsort(0)
    nf = len(face_inds)
    ndist = len(bo_inds)
    roc_FSI = (R[:nf].sum(0) - ((nf)**2 + nf)/2.) / float(nf*ndist)
    roc_FSI_counts = [len((roc_FSI > thres).nonzero()[0]) for thres in thresholds]
    roc_FSI_fractions = [c/ float(num_features) for c in roc_FSI_counts]
    record['roc_fsi_fractions'] = roc_FSI_fractions
    record['roc_FSI_s_avg'] = roc_FSI.mean(2).tolist()
    record['roc_Face_selective_s_avg'] = (roc_FSI > .75).astype(np.float).mean(2).tolist()
    
    if train:
        features = features.reshape((fs[0],num_features))
        STATS = ['train_accuracy','train_ap','train_auc','test_accuracy','test_ap','test_auc']
        catfuncs = [('Face_Nonface',lambda x : x['name'] if x['name'] == 'Face' else 'Nonface'),
                    ('Face_Body_Object',None)]
        record['training_data'] = {}
        for (problem, func) in catfuncs:
            results = traintest(dataset, features, catfunc=func, **training_data)
            stats = {}
            for stat in STATS:
                stats[stat] = np.mean([r[1][stat] for r in results])
            record['training_data'][problem] = stats
    return record, FSI


############################
########Original FBO Bandits
############################

class SimffaBandit(gb.GensonBandit):

    def __init__(self):
        super(SimffaBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl):
        dataset = skdata.fbo.FaceBodyObject20110803() 
        record, FSI = evaluate_FSI(dataset, config)
        record['loss'] = 1 - (record['training_data']['Face_Nonface']['test_accuracy'])/100.
        print('DONE')
        return record


class SimffaL1Bandit(SimffaBandit):
    source_string = gh.string(simffa_params.l1_params)


class SimffaL1GaborBandit(SimffaBandit):
    source_string = gh.string(simffa_params.l1_params_gabor)


class SimffaL1GaborLargerBandit(SimffaBandit):
    source_string = gh.string(simffa_params.l1_params_gabor_larger)


class SimffaL2Bandit(SimffaBandit):
    source_string = gh.string(simffa_params.l2_params)


class SimffaL2GaborBandit(SimffaBandit):
    source_string = gh.string(simffa_params.l2_params_gabor)


class SimffaL3Bandit(SimffaBandit):
    source_string = gh.string(simffa_params.l3_params)


class SimffaL3GaborBandit(SimffaBandit):
    source_string = gh.string(simffa_params.l1_params_gabor)



#########################
########Invariant bandits
#########################

class SimffaInvariantBandit(gb.GensonBandit):
    training_data = {}
    train = True
    new_backgrounds = False
    
    def __init__(self):
        super(SimffaInvariantBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl):
        dataset = skdata.fbo.FaceBodyObject20110803() 
        original_record, FSI = evaluate_FSI(dataset, config, train=cls.train, **cls.training_data)
        if cls.new_backgrounds:
            invariant_dataset0 = fbo_invariant.FaceBodyObject20110803Invariant0_b() 
            invariant_dataset1 = fbo_invariant.FaceBodyObject20110803Invariant1_b() 
            invariant_dataset2 = fbo_invariant.FaceBodyObject20110803Invariant2_b() 
            invariant_dataset_flip = fbo_invariant.FaceBodyObject20110803InvariantFlip_b() 
        else:
            invariant_dataset0 = fbo_invariant.FaceBodyObject20110803Invariant0() 
            invariant_dataset1 = fbo_invariant.FaceBodyObject20110803Invariant1() 
            invariant_dataset2 = fbo_invariant.FaceBodyObject20110803Invariant2() 
            invariant_dataset_flip = fbo_invariant.FaceBodyObject20110803InvariantFlip() 
        invariant_record0, FSI = evaluate_FSI(invariant_dataset0, config, train=cls.train, **cls.training_data)
        invariant_record1, FSI = evaluate_FSI(invariant_dataset1, config, train=cls.train, **cls.training_data)
        invariant_record2, FSI = evaluate_FSI(invariant_dataset2, config, train=cls.train, **cls.training_data)
        invariant_record_flip, FSI = evaluate_FSI(invariant_dataset_flip, config, train=cls.train, **cls.training_data)            
        record = {'original': original_record, 
                  'invariant0': invariant_record0,
                  'invariant1': invariant_record1,
                  'invariant2': invariant_record2,
                  'invariant_flip': invariant_record_flip}
        Ks = [_x + _y for _y in ['FSI_s_avg', 'Face_selective_s_avg'] for _x in ['','shifted_', 'rectified_']] + \
             ['F_s_avg', 'BO_s_avg', 'dprime_selective_s_avg','roc_FSI_s_avg', 'roc_Face_selective_s_avg']
        if hasattr(ctrl, 'attachments'):
            for k in record:
                for l in Ks:
                    spatial_averages = record[k].pop(l)
                    blob = cPickle.dumps(spatial_averages)
                    ctrl.attachments['spatial_averages_' + k + '_' + l] = blob
        if cls.train:
            record['loss'] = 1 - (record['invariant1']['training_data']['Face_Nonface']['test_accuracy'])/100.
        else:
            record['loss'] = 1
        print('DONE')
        return record


class SimffaL1InvariantBandit(SimffaInvariantBandit):
    training_data = {'num_splits': 3}
    source_string = gh.string(simffa_params.l1_params)


class SimffaL2InvariantBandit(SimffaInvariantBandit):
    source_string = gh.string(simffa_params.l2_params)


class SimffaL3InvariantBandit(SimffaInvariantBandit):
    source_string = gh.string(simffa_params.l3_params)


class SimffaPixelsInvariantBandit(SimffaInvariantBandit):
    source_string = gh.string(simffa_params.pixels_params)


class SimffaV1LikeInvariantBandit(SimffaInvariantBandit):
    training_data = {'num_splits': 3}
    source_string = gh.string(simffa_params.v1like_params)
    
    
class SimffaV1LikeSpectrumInvariantBandit(SimffaInvariantBandit):
    training_data = {'num_splits': 3}
    source_string = gh.string(simffa_params.v1like_spectrum_params)
   
    
class SimffaL1InvariantBanditNew(SimffaInvariantBandit):
    new_backgrounds = True
    train = False
    source_string = gh.string(simffa_params.l1_params)


class SimffaL2InvariantBanditNew(SimffaInvariantBandit):
    new_backgrounds =True
    train = False
    source_string = gh.string(simffa_params.l2_params)


class SimffaL3InvariantBanditNew(SimffaInvariantBandit):
    new_backgrounds = True
    train = False
    source_string = gh.string(simffa_params.l3_params)


class SimffaPixelsInvariantBanditNew(SimffaInvariantBandit):
    new_backgrounds = True
    train = False
    source_string = gh.string(simffa_params.pixels_params)


class SimffaV1LikeInvariantBanditNew(SimffaInvariantBandit):
    new_backgrounds = True
    train = False
    source_string = gh.string(simffa_params.v1like_params)



########################
########Facelike bandits
########################

def evaluate_facelike(config, credentials, FSI=None):
    dataset = simffa_datasets.Facelike(credentials)
    X, labels = dataset.img_regression_task()
    all_paths, labels1 = dataset.raw_regression_task()
    assert (labels == labels1).all()
    assert (all_paths == sorted(all_paths)).all()
    features = get_features(X, config, verbose=True)
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]
    if FSI is not None:
        FSI_shape = FSI.shape
        assert FSI_shape == fs[1:]
        FSI = np.ravel(FSI)

    record = {}
    record['num_features'] = num_features
    record['feature_shape'] = fs

    subjects = [('subject_avg','avg')] # + [('subject_' + str(ind),ind) for ind in range(5)]

    bins = np.arange(-1, 1, .01)
    record['bins'] = bins.tolist()
    for subject, judgement in subjects:
        record[subject] = {}
        for name, subset in dataset.SUBSETS + [('all',None)]:
            print('Evaluating', subject, name)
            subpaths, sublabels = dataset.raw_regression_task(subset=subset,
                                                        judgement=judgement)
            inds = all_paths.searchsorted(subpaths)
            f_subset = features[inds]
            label_subset = labels[inds]
            P, P_prob = pearsonr(f_subset, label_subset)
            P = np.ma.masked_array(P, np.isnan(P))
            
            f_rshp = f_subset.reshape((fs[0],num_features))
            P_pop, P_pop_prob = sp_stats.pearsonr(f_rshp.mean(1), label_subset)
            S_pop, S_pop_prob = sp_stats.spearmanr(f_rshp.mean(1), label_subset)

            data = {}
            data['Pearson_hist'] = np.histogram(P, bins)[0].tolist()
            data['Pearson_avg'] = float(P.mean())
            data['Pearson_s_avg'] = P.mean(2).tolist()
            data['Pearson_pop_avg'] = float(P_pop)
            data['Spearman_pop_avg'] = float(S_pop)
            
            if FSI is not None:
                assert FSI_shape == P.shape == f_subset.shape[1:]
                sel_inds = (FSI > 1./3)
                data['Pearson_pop_avg_sel'] = float(sp_stats.pearsonr(f_rshp[sel_inds].mean(1), label_subset))
                data['Spearman_pop_avg_sel'] = float(sp_stats.spearmanr(f_rshp[sel_inds].mean(1), label_subset))
                P = P.ravel()
                sel_inds = np.invert(np.isnan(P))
                data['Pearson_FSI_corr'] = np.corrcoef(FSI[sel_inds], P[sel_inds]).tolist()
                sel_inds = (FSI > 1./3) & np.invert(np.isnan(P))
                data['Pearson_hist_sel'] = np.histogram(P[sel_inds], bins)[0].tolist()
                data['Pearson_avg_sel'] = float(P[sel_inds].mean())
                data['Pearson_FSI_corr_sel'] = np.corrcoef(FSI[sel_inds], P[sel_inds]).tolist()
 
            record[subject][name] = data

    return record


class SimffaFacelikeBandit(gb.GensonBandit):
    """
    call with bandit-argfile supplying credentials
    """
    def __init__(self, credentials):
        super(SimffaFacelikeBandit, self).__init__(source_string=self.source_string)
        self.credentials = tuple(credentials)

    def evaluate(self, config, ctrl):
        record = {}
        dataset = skdata.fbo.FaceBodyObject20110803()
        FSI_rec, FSI = evaluate_just_FSI(dataset, config, train=False)
        record['FSI'] = FSI_rec
        record['Facelike'] = evaluate_facelike(config, self.credentials, FSI=FSI)
        record['loss'] = .5 * (1 - record['Facelike']['subject_avg']['all']['Pearson_avg'])
        print('DONE')

        return record


class SimffaFacelikeL1Bandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l1_params)


class SimffaFacelikeL2Bandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l2_params)


class SimffaFacelikeL3Bandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l3_params)


class SimffaFacelikeL1GaborBandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l1_params_gabor)


class SimffaFacelikeL1GaborLargerBandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l1_params_gabor_larger)


class SimffaFacelikeL2GaborBandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l2_params_gabor)


class SimffaFacelikeL3GaborBandit(SimffaFacelikeBandit):
    source_string = gh.string(simffa_params.l1_params_gabor)


                            
