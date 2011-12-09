import copy

import pymongo as pm
import numpy as np
from thoreano.slm import TheanoSLM
import skdata.fbo
import hyperopt.genson_bandits as gb

import simffa_params
from classifier import train_liblinear_classifier
from theano_slm import slm_from_config, FeatureExtractor


def get_features(dataset, config):
	X, y = dataset.img_classification_task()
	batchsize = 4
	slm = slm_from_config(config, X.shape, batchsize=batchsize, use_theano=True)
	extractor = FeatureExtractor(X, slm, batchsize=batchsize)
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
        result = train_liblinear_classifier(train_Xy, test_Xy)
        results.append(result)
    return results


class SimffaBandit(gb.GensonBandit):

    def __init__(self):
        super(SimffaBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl):

        dataset = skdata.fbo.FaceBodyObject20110803()
        features = get_features(dataset, config)
        fs = features.shape
        num_features = fs[1]*fs[2]*fs[3]
        
        record = {}
        thresholds = np.arange(0,1,.01)
        F = features[:20].mean(0)
        BO = features[20:].mean(0)
        FSI = (F - BO) / (np.abs(F) + np.abs(BO))
        FSI_counts = [len((FSI > thres).nonzero()[0]) for thres in thresholds]
        FSI_fractions = [c/ float(num_features) for c in FSI_counts]
        record['num_features'] = num_features
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
                stats[stat] = np.mean([r[2][stat] for r in results])
            record['training_data'][problem] = stats

        record['loss'] = 1 - (record['training_data']['Face_Nonface']['test_accuracy'])/100.

        return record


class SimffaL1Bandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l1_params)


class SimffaL1GaborBandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l1_params_gabor)


class SimffaL1GaborLargerBandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l1_params_gabor_larger)


class SimffaL2Bandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l2_params)


class SimffaL2GaborBandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l1_params_gabor)


class SimffaL3Bandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l3_params)


class SimffaL3GaborBandit(SimffaBandit):
    source_string = simffa_params.string(simffa_params.l1_params_gabor)


def make_plots():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    
    exp_key1 = 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key2 = 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key3 = 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key1g = 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom'
    
    C1 = list(Jobs.find({'exp_key':exp_key1,'state':2}))
    C2 = list(Jobs.find({'exp_key':exp_key2,'state':2}))
    C3 = list(Jobs.find({'exp_key':exp_key3,'state':2}))
    C1g = list(Jobs.find({'exp_key':exp_key1g,'state':2}))

    FSI_1 = np.array([c['result']['fsi_fractions'] for c in C1])
    FSI_2 = np.array([c['result']['fsi_fractions'] for c in C2])
    FSI_3 = np.array([c['result']['fsi_fractions'] for c in C3])
    FSI_1g = np.array([c['result']['fsi_fractions'] for c in C1g])
    
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.plot(FSI_1.mean(0))
    plt.plot(FSI_2.mean(0))
    plt.plot(FSI_3.mean(0))
    plt.plot(FSI_1g.mean(0))
    plt.legend(('L1','L2','L3','L1 Gabor'))
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
    
def make_plots2():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    
    exp_key1 = 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key2 = 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key3 = 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom'
    exp_key1g = 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom'
    
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