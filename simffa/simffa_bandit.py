"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy

import pymongo as pm
import numpy as np
import scipy.stats as sp_stats

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
from thoreano.slm import slm_from_config, FeatureExtractor
from thoreano.classifier import train_scikits
import skdata.fbo

import simffa_params
import simffa_datasets
from simffa_stats import pearsonr, spearmanr


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
    for ind in range(5):
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
    

def evaluate_just_FSI(dataset, config):
    record, FSI = evaluate_FSI(dataset, config, train=False)
    for k in ['F_s_avg', 'BO_s_avg', 'FSI_s_avg']:
        record.pop(k)       
    return record, FSI


def evaluate_FSI(dataset, config, train=True):
    X, y = dataset.img_classification_task()
    features = get_features(X, config, verbose=True)
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]
    record = {}
    record['num_features'] = num_features
    record['feature_shape'] = fs
    thresholds = np.arange(0,1,.01)
    meta = dataset.meta
    face_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] == 'Face']
    bo_inds = [ind for ind in range(len(meta)) if meta[ind]['name'] != 'Face']
    F = features[face_inds].mean(0)
    BO = features[bo_inds].mean(0)
    FSI = (F - BO) / (np.abs(F) + np.abs(BO))
    FSI_counts = [len((FSI > thres).nonzero()[0]) for thres in thresholds]
    FSI_fractions = [c/ float(num_features) for c in FSI_counts]
    record['thresholds'] = thresholds.tolist()
    record['fsi_fractions'] = FSI_fractions
    record['F_s_avg'] = F.mean(2).tolist()
    record['BO_s_avg'] = BO.mean(2).tolist()
    record['FSI_s_avg'] = FSI.mean(2).tolist()
    record['Face_selective_s_avg'] = (FSI > .333).astype(np.float).mean(2).tolist()
    if train:
        features = features.reshape((fs[0],num_features))
        STATS = ['train_accuracy','train_ap','train_auc','test_accuracy','test_ap','test_auc']
        catfuncs = [('Face_Nonface',lambda x : x['name'] if x['name'] == 'Face' else 'Nonface'),
                    ('Face_Body_Object',None)]
        record['training_data'] = {}
        for (problem, func) in catfuncs:
            results = traintest(dataset, features, catfunc=func)
            stats = {}
            for stat in STATS:
                stats[stat] = np.mean([r[1][stat] for r in results])
            record['training_data'][problem] = stats
    return record, FSI


class SimffaBandit(gb.GensonBandit):

    def __init__(self):
        super(SimffaBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl):
        dataset = skdata.fbo.FaceBodyObject20110803() 
        record, FSI = evalute_FSI(dataset, config)
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
        FSI_rec, FSI = evaluate_just_FSI(dataset, config)
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


                            
