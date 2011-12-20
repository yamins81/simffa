"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy

import pymongo as pm
import numpy as np
import skdata.fbo
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
from thoreano.slm import slm_from_config, FeatureExtractor
from thoreano.classifier import train_scikits

import simffa_params
from stats import pearsonr, spearmanr

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


def evaluate_FSI(config):
        dataset = skdata.fbo.FaceBodyObject20110803()
        X, y = dataset.img_classification_task()
        features = get_features(X, config)
        fs = features.shape
        num_features = fs[1]*fs[2]*fs[3]

        record = {}
        record['num_features'] = num_features
        record['feature_shape'] = fs

        thresholds = np.arange(0,1,.01)
        F = features[:20].mean(0)
        BO = features[20:].mean(0)
        FSI = (F - BO) / (np.abs(F) + np.abs(BO))
        FSI_counts = [len((FSI > thres).nonzero()[0]) for thres in thresholds]
        FSI_fractions = [c/ float(num_features) for c in FSI_counts]
        record['thresholds'] = thresholds.tolist()
        record['fsi_fractions'] = FSI_fractions
        record['F_s_avg'] = F.mean(2).tolist()
        record['BO_s_avg'] = BO.mean(2).tolist()
        record['FSI_s_avg'] = FSI.mean(2).tolist()
        record['Face_selective_s_avg'] = (FSI > .333).astype(np.float).mean(2).tolist()

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
        record, FSI = evalute_FSI(config)
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


def make_plots():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key1 = 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key2 = 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key3 = 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key1g = 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'
    exp_key2g = 'simffa.simffa_bandit.SimffaL2GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'


    C1 = list(Jobs.find({'exp_key':exp_key1,'state':2}))
    C2 = list(Jobs.find({'exp_key':exp_key2,'state':2}))
    C3 = list(Jobs.find({'exp_key':exp_key3,'state':2}))
    C1g = list(Jobs.find({'exp_key':exp_key1g,'state':2}))
    C2g = list(Jobs.find({'exp_key':exp_key1g,'state':2}))

    FSI_1 = np.array([c['result']['fsi_fractions'] for c in C1])
    FSI_2 = np.array([c['result']['fsi_fractions'] for c in C2])
    FSI_3 = np.array([c['result']['fsi_fractions'] for c in C3])
    FSI_1g = np.array([c['result']['fsi_fractions'] for c in C1g])
    FSI_2g = np.array([c['result']['fsi_fractions'] for c in C2g])

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.plot(FSI_1.mean(0))
    plt.plot(FSI_2.mean(0))
    plt.plot(FSI_3.mean(0))
    plt.plot(FSI_1g.mean(0))
    plt.plot(FSI_2g.mean(0))
    plt.legend(('L1','L2','L3','L1 Gabor','L2 Gabor'))
    plt.ylabel('Fraction of taps with FSI > threshold')
    plt.xlabel('Threshold')
    plt.title('FSI fractions vs. threshold, averaged over models')
    plt.savefig('Averages.png')

    plt.figure()
    plt.boxplot([FSI_1[:,l*10] for l in range(10)])
    plt.plot(range(1,11),FSI_1.mean(0)[[l*10 for l in range(10)]],color='green')
    plt.scatter(range(1,11),FSI_1.mean(0)[[l*10 for l in range(10)]])
    plt.title('L1 boxplot -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.savefig('L1_boxplot.png')
    plt.figure()
    plt.boxplot([FSI_2[:,l*10] for l in range(10)])
    plt.plot(range(1,11),FSI_2.mean(0)[[l*10 for l in range(10)]], color='green')
    plt.scatter(range(1,11),FSI_2.mean(0)[[l*10 for l in range(10)]])
    plt.title('L2 boxplot -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.savefig('L2_boxplot.png')
    plt.figure()
    plt.boxplot([FSI_3[:,l*10] for l in range(10)])
    plt.plot(range(1,11),FSI_3.mean(0)[[l*10 for l in range(10)]], color='green')
    plt.scatter(range(1,11),FSI_3.mean(0)[[l*10 for l in range(10)]])
    plt.title('L3 boxplot -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.savefig('L3_boxplot.png')
    plt.figure()
    plt.boxplot([FSI_1g[:,l*10] for l in range(10)])
    plt.plot(range(1,11),FSI_1g.mean(0)[[l*10 for l in range(10)]], color='green')
    plt.scatter(range(1,11),FSI_1g.mean(0)[[l*10 for l in range(10)]])
    plt.title('L1g boxplot -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.savefig('L1g_boxplot.png')
    plt.figure()
    plt.boxplot([FSI_2g[:,l*10] for l in range(10)])
    plt.plot(range(1,11),FSI_2g.mean(0)[[l*10 for l in range(10)]], color='green')
    plt.scatter(range(1,11),FSI_2g.mean(0)[[l*10 for l in range(10)]])
    plt.title('L2g boxplot -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.savefig('L2g_boxplot.png')


def make_plots2():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key1 = 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key2 = 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key3 = 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key1g = 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'

    C1 = list(Jobs.find({'exp_key':exp_key1,'state':2}))
    C2 = list(Jobs.find({'exp_key':exp_key2,'state':2}))
    C3 = list(Jobs.find({'exp_key':exp_key3,'state':2}))
    C1g = list(Jobs.find({'exp_key':exp_key1g,'state':2}))

    import matplotlib.pyplot as plt
    plt.ioff()

    X1 = np.array([(c['result']['fsi_fractions'][10],c['result']['training_data']['Face_Nonface']['test_accuracy']) for c in C1])
    plt.figure()
    plt.plot(X1[:,0],X1[:,1],'o')
    plt.title('L1 Model performance on Face/nonface vs FSI fraction at threshold .1')
    plt.xlabel('FSI Fraction at threshold .1')
    plt.ylabel('Model Performance (test accuracy, ntrain=ntest=10, 5 splits)')
    plt.savefig('L1_performance_vs_FSI.png')
    X2 = np.array([(c['result']['fsi_fractions'][10],c['result']['training_data']['Face_Nonface']['test_accuracy']) for c in C1])
    plt.figure()
    plt.plot(X2[:,0],X2[:,1],'o')
    plt.title('L2 Model performance on Face/nonface vs FSI fraction at threshold .1')
    plt.xlabel('FSI Fraction at threshold .1')
    plt.ylabel('Model Performance (test accuracy, ntrain=ntest=10, 5 splits)')
    plt.savefig('L2_performance_vs_FSI.png')

    X3 = np.array([(c['result']['fsi_fractions'][10],c['result']['training_data']['Face_Body_Object']['test_accuracy']) for c in C3])
    plt.figure()
    plt.plot(X3[:,0],X3[:,1],'o')
    plt.title('L3 Model performance on Face/Body/Object vs FSI fraction at threshold .1')
    plt.xlabel('FSI Fraction at threshold .1')
    plt.ylabel('Model Performance (test accuracy, ntrain=ntest=10, 5 splits)')
    plt.savefig('L3_FBO_performance_vs_FSI.png')

    X4 = np.array([(c['result']['fsi_fractions'][10],c['result']['training_data']['Face_Body_Object']['test_accuracy']) for c in C1])
    plt.figure()
    plt.plot(X4[:,0],X4[:,1],'o')
    plt.title('L1 Model performance on Face/Body/Object vs FSI fraction at threshold .1')
    plt.xlabel('FSI Fraction at threshold .1')
    plt.ylabel('Model Performance (test accuracy, ntrain=ntest=10, 5 splits)')
    plt.savefig('L1_FBO_performance_vs_FSI.png')

import os
def make_plots3():

    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    ekeys = [('L1', 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2', 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L3', 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L1g', 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2g', 'simffa.simffa_bandit.SimffaL2GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial')]

    import matplotlib.pyplot as plt

    for (n,k) in ekeys:
        os.mkdir(n + '_Face_selective')
        C  = list(Jobs.find({'exp_key':k,'state':2}))
        for (ind,c) in enumerate(C):
            A = np.array(c['result']['Face_selective_s_avg'])
            plt.imshow(A)
            plt.savefig(n + '_Face_selective/' + str(ind) + '.png')


import os
def make_plots4():

    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    ekeys = [('L1', 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2', 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L3', 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L1g', 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2g', 'simffa.simffa_bandit.SimffaL2GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial')]

    import matplotlib.pyplot as plt

    for (n,k) in ekeys:
        os.mkdir(n + '_Faces_avg')
        C  = list(Jobs.find({'exp_key':k,'state':2}))
        for (ind,c) in enumerate(C):
            A = np.array(c['result']['F_s_avg'])
            plt.imshow(A)
            plt.savefig(n + '_Faces_avg/' + str(ind) + '.png')


def compute_blobiness():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    ekeys = [('L1', 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2', 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L3', 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L1g', 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2g', 'simffa.simffa_bandit.SimffaL2GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial')]

    import matplotlib.pyplot as plt
    things = []
    func = lambda x : -np.log(np.abs((x - x.mean())/len(x.ravel())).mean())
    for (n,k) in ekeys:
        C  = list(Jobs.find({'exp_key':k,'state':2}))
        A = [np.array(c['result']['Face_selective_s_avg']) for c in C]
        B = map(lambda x : func(x),filter(lambda x : (x > 0).any(),A))
        things.append(B)

    plt.figure()
    plt.boxplot(things)
    plt.plot(range(1,6),[np.mean(t) for t in things],color='green')
    plt.scatter(range(1,6),[np.mean(t) for t in things])
    plt.title('Inverse Peakiness -- mean shown in green')
    plt.xlabel('Threshold')
    plt.ylabel('FSI at threshold')
    plt.xticks(range(1,6), zip(*ekeys)[0])
    plt.savefig('Inverse_Peakiness.png')

    return things



######facelike regression

import facelike

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


def evaluate_facelike(config, credentials, FSI=None):
    dataset = facelike.Facelike(credentials)
    X, labels = dataset.img_regression_task()
    all_paths, labels1 = dataset.raw_regression_task()
    assert (labels == labels1).all()
    assert (all_paths == sorted(all_paths)).all()
    features = get_features(X, config)
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
            subpaths, sublabels = dataset.raw_regression_task(subset=subset,
                                                        judgement=judgement)
            inds = all_paths.searchsorted(subpaths)
            f_subset = features[inds]
            label_subset = labels[inds]
            P, P_prob = pearsonr(f_subset, label_subset)
            S, S_prob = spearmanr(f_subset, label_subset)
            P = np.ma.masked_array(P,np.isnan(P))
            S = np.ma.masked_array(S,np.isnan(S))

            data = {}
            data['Pearson_hist'] = np.histogram(P, bins)[0].tolist()
            data['Pearson_avg'] = P.mean()
            data['Spearman_hist'] = np.histogram(S, bins)[0].tolist()
            data['Spearman_avg'] = S.mean()
            data['Pearson_s_avg'] = P.mean(2).tolist()
            data['Spearman_s_avg'] = S.mean(2).tolist()
            if FSI is not None:
                assert FSI_shape == P.shape == S.shape
                P = P.ravel()
                sel_inds = np.invert(np.isnan(P))
                data['Pearson_FSI_corr'] = np.corrcoef(FSI[sel_inds], P[sel_inds])
                S = S.ravel()
                sel_inds = np.invert(np.isnan(S))
                data['Spearman_FSI_corr'] = np.corrcoef(FSI[sel_inds], S[sel_inds])
                sel_inds = (FSI > 1./3) & np.invert(np.isnan(P))
                data['Pearson_hist_sel'] = np.histogram(P[sel_inds], bins)[0].tolist()
                data['Pearson_avg_sel'] = P[sel_inds].mean()
                data['Pearson_FSI_corr_sel'] = np.corrcoef(FSI[sel_inds], P[sel_inds])
                sel_inds = (FSI > 1./3)& np.invert(np.isnan(S))
                data['Spearman_hist_sel'] = np.histogram(S[sel_inds], bins)[0].tolist()
                data['Spearman_avg_sel'] = S[sel_inds].mean()
                data['Spearman_FSI_corr_sel'] = np.corrcoef(FSI[sel_inds], S[sel_inds])

            record[subject][name] = data

        #r_cp = copy.deepcopy(record[subject])
        #r_cp.pop('all')
        #record[subject]['subset_avg'] = dict_avg(r_cp)

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
        FSI_rec, FSI = evaluate_FSI(config)
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


def dict_avg(d):
    e = {}
    K = d.keys()
    KK = d[K].keys()
    for kk in KK:
        e[k] = float(np.mean([d[k][kk] for k in K]))
    return e

