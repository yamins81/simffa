import os

import pymongo as pm
import numpy as np
import tabular as tb
import scipy.signal as signal


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


def Var(x):
    #return np.abs(x - x.mean()).mean()
    return x.var()
    
def compute_clusteriness():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    ekeys = [('L1', 'simffa.simffa_bandit.SimffaL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L2', 'simffa.simffa_bandit.SimffaL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    ('L3', 'simffa.simffa_bandit.SimffaL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    #('L1g', 'simffa.simffa_bandit.SimffaL1GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial'),
    #('L2g', 'simffa.simffa_bandit.SimffaL2GaborBandit/hyperopt.theano_bandit_algos.TheanoRandom/spatial')
    ]

    Frac = 10.
    arrs = []
    for (label,k) in ekeys:
        print('doing', label)
        C  = Jobs.find({'exp_key':k,'state':2})
        recs = []
        for (ind,c) in enumerate(C):
            print(label, ind)
            nf = c['result']['num_features']
            A = np.array(c['result']['Face_selective_s_avg'])
            av = Var(A)
            if av > 0:
                am = A.mean()
                amax = A.max()
                nl = nf/(A.shape[0]*A.shape[1])
                sh = (s0,s1) =  (A.shape[0]/Frac, A.shape[1]/Frac)
                Ac = signal.convolve2d(A,np.ones(sh),mode='same') / (s0*s1)
                Ac = Ac[s0/2+1:-(s0/2+1),s1/2+1:-(s1/2+1)]
                acv = Var(Ac)
                arat = av/acv
                r = (np.random.random((nf,)) < c['result']['fsi_fractions'][33]).astype(np.int).reshape(A.shape + (nl,)).mean(2)
                rv = Var(r)
                rm = r.mean()
                rmax = r.max()
                rc = signal.convolve2d(r,np.ones(sh),mode='same') / (s0*s1)
                rc = rc[s0/2+1:-(s0/2+1),s1/2+1:-(s1/2+1)]
                rcv = Var(rc)
                rrat = rv/rcv
                arrat = arat/rrat
                recs.append((ind,A.shape[0], A.shape[1], av,am,amax,acv,arat,rv,rm,rmax,rcv,rrat,arrat))
                
        X = tb.tabarray(records=recs, names=['ind','s0','s1','av','am','amax','acv','arat','rv','rm','rmax','rcv','rrat','arrat'])
        arrs.append(X) 
            
    return arrs


def plot_clusteriness(arrs):
    import matplotlib.pyplot as plt
    plt.figure()
    
    B = plt.boxplot([(1/a['rrat']) for a in arrs])
    B1 = plt.boxplot([(1/a['arat']) for a in arrs])
    for b in B['boxes']:
        b.set_color('g')
    plt.draw()
    
    plt.plot(range(1,4),[np.mean(1/a['arat']) for a in arrs],color='blue')
    plt.scatter(range(1,4),[np.mean(1/a['arat']) for a in arrs],color='blue')
    plt.plot(range(1,4),[np.mean(1/a['rrat']) for a in arrs],color='green')
    plt.scatter(range(1,4),[np.mean(1/a['rrat']) for a in arrs],color='green')
    
    plt.title('Spatial clustering measure:\nActual (blue) vs random mean/var-matched (green)')
    plt.xticks(range(1,4),('L1','L2','L3'))
    plt.ylabel('Clustering measure')
    plt.savefig('Clustering.png')


def make_facelike_spatial_plots():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    import matplotlib.pyplot as plt
    ekeys = [
             ('L1', u'simffa.simffa_bandit.SimffaFacelikeL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L2', u'simffa.simffa_bandit.SimffaFacelikeL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L3', u'simffa.simffa_bandit.SimffaFacelikeL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]')
            ]
    
    subsets = map(str,[u'all',  u'eye', u'nose', u'eye-mouth', u'eye-nose', u'eye-eye'])
    fields = ['result.Facelike.subject_avg.' + subset + '.Pearson_s_avg' for subset in subsets]
    
    for label, e in ekeys:    
        H = Jobs.find({'exp_key':e,'state':2}, fields=fields, timeout=False)
        cnt = Jobs.find({'exp_key':e,'state':2}).count()
        dir = os.path.join(label + '_pearson_s_average')
        os.makedirs(dir)
        plt.subplots_adjust(hspace=.5)
        for (_i,h) in enumerate(H):
            print('%s: %d of %d' % (label, _i, cnt))    
            for (ind, subset) in enumerate(subsets):
                p = plt.subplot(3, 2, ind+1)
                v = h['result']['Facelike']['subject_avg'][subset]['Pearson_s_avg']
                correct(v)
                p.imshow(v)
                p.set_title(subset)
            plt.draw()
            path = os.path.join(dir, str(_i) + '_spatial.png')
            plt.savefig(path)
            plt.clf()
            

def make_facelike_FSI_Corr():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    import matplotlib.pyplot as plt
    ekeys = [
             ('L1', u'simffa.simffa_bandit.SimffaFacelikeL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L2', u'simffa.simffa_bandit.SimffaFacelikeL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L3', u'simffa.simffa_bandit.SimffaFacelikeL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]')
            ]
    
    subsets = map(str,[u'all',  u'eye', u'nose', u'eye-mouth', u'eye-nose', u'eye-eye'])
    fields = ['result.Facelike.subject_avg.' + subset + '.' + x for subset in subsets for x in ['Pearson_FSI_corr', 'Pearson_FSI_corr_sel']]
    
    res = []
    res_sel = []
    for label, e in ekeys:    
        H = list(Jobs.find({'exp_key':e,'state':2}, fields=fields, timeout=False))
        col = []
        col_sel = []
        for subset in subsets:
            x = np.array([h['result']['Facelike']['subject_avg'][subset]['Pearson_FSI_corr'][0][1] for h in H])
            x[np.isnan(x)] = 0
            col.append(x.mean())
            
            y = np.array([h['result']['Facelike']['subject_avg'][subset]['Pearson_FSI_corr_sel'][0][1] for h in H])
            y[np.isnan(y)] = 0
            col_sel.append(y.mean())
        res.append(col)
        res_sel.append(col_sel)
    
    X = tb.tabarray(columns = [subsets] + res, names = ('subset',) + zip(*ekeys)[0])
    X_sel = tb.tabarray(columns = [subsets] + res_sel, names = ('subset',) + zip(*ekeys)[0])
    
    import matplotlib.pyplot as plt
    for x in X[['L1','L2','L3']].extract():
        plt.plot(x)
        plt.scatter(range(len(x)),x)
    plt.xticks((0,1,2),('L1','L2','L3'))
    plt.legend(tuple(X['subset']), loc='best')
    plt.ylabel('Mean pearson-FSI Correlation over all models')
    plt.savefig('pearson_FSI_correlation.png')
            
    return X, X_sel
    
def make_facelike_FSI_Corr_boxplots():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    import matplotlib.pyplot as plt
    ekeys = [
             ('L1', u'simffa.simffa_bandit.SimffaFacelikeL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L2', u'simffa.simffa_bandit.SimffaFacelikeL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L3', u'simffa.simffa_bandit.SimffaFacelikeL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]')
            ]
    
    subsets = map(str,[u'all',  u'eye', u'nose', u'eye-mouth', u'eye-nose', u'eye-eye'])
    fields = ['result.Facelike.subject_avg.' + subset + '.' + x for subset in subsets for x in ['Pearson_FSI_corr', 'Pearson_FSI_corr_sel']]
    
    rows = dict([(x,[]) for x in subsets])

    for label, e in ekeys:    
        H = list(Jobs.find({'exp_key':e,'state':2}, fields=fields, timeout=False))
        for subset in subsets:
            x = np.array([h['result']['Facelike']['subject_avg'][subset]['Pearson_FSI_corr'][0][1] for h in H])
            x = x[np.invert(np.isnan(x))]
            rows[subset].append(x)
    
    plt.figure(figsize=(15,15))
    for (s_ind, subset) in enumerate(subsets):
        p = plt.subplot(3,2,s_ind+1)
        p.boxplot(rows[subset])
        p.plot(range(1,4),[m.mean() for m in rows[subset]],color='g')
        p.scatter(range(1,4),[m.mean() for m in rows[subset]],color='g')
        p.set_xticks(range(1,4))
        p.set_xticklabels(('L1','L2','L3'))

        if p.colNum == 0:
            p.set_ylabel('Facelike/FSI fit')
        else:
            p.yaxis.set_visible(False)
        p.set_title(subset,x=.5,y=1)
        
        #p.set_ylabel('Mean pearson-FSI Correlation over all models')
    
    plt.subplots_adjust(wspace=.05, hspace=.15)
    plt.suptitle('Facelike-FSI Correlation over all models', fontsize=20)
    plt.draw()
    plt.savefig('pearson_FSI_correlation_boxplots.png')
            
    return rows
        
                
def make_facelike_hists():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    import matplotlib.pyplot as plt
    ekeys = [('L1', u'simffa.simffa_bandit.SimffaFacelikeL1Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L2', u'simffa.simffa_bandit.SimffaFacelikeL2Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]'),
             ('L3', u'simffa.simffa_bandit.SimffaFacelikeL3Bandit/hyperopt.theano_bandit_algos.TheanoRandom[arghash:39db382c5f3ba8dd0c8af1864455e618]')]
    
    subsets = map(str,[u'all',  u'eye', u'nose', u'eye-mouth', u'eye-nose', u'eye-eye'])
    fields = ['result.Facelike.bins'] + ['result.Facelike.subject_avg.' + subset + '.' + x for subset in subsets for x in ['Pearson_hist', 'Pearson_hist_sel']]
    
    for label, e in ekeys:    
        H = Jobs.find({'exp_key':e,'state':2}, fields=fields, timeout=False)
        cnt = Jobs.find({'exp_key':e,'state':2}).count()
        dir = os.path.join(label + '_hists')
        os.makedirs(dir)
        plt.subplots_adjust(hspace=.5)
        for (_i,h) in enumerate(H):
            print('%s: %d of %d' % (label, _i, cnt))
            bins = h['result']['Facelike']['bins']            
            for (ind, subset) in enumerate(subsets):
                p = plt.subplot(3, 2, ind+1)
                v = np.array(h['result']['Facelike']['subject_avg'][subset]['Pearson_hist'])
                if v.sum() > 0:
                    v = v/float(v.sum())
                p.bar(bins[:-1], v, width=bins[1]-bins[0])
                p.set_title(subset)
            plt.draw()
            path = os.path.join(dir, str(_i) + '_hist.png')
            plt.savefig(path)
            plt.clf()
            for (ind, subset) in enumerate(subsets):
                p = plt.subplot(3, 2, ind+1)
                v = np.array(h['result']['Facelike']['subject_avg'][subset]['Pearson_hist_sel'])
                if v.sum() > 0:
                    v = v/float(v.sum())
                p.bar(bins[:-1], v, width=bins[1]-bins[0])
                p.set_title(subset)
            plt.draw()
            path = os.path.join(dir, str(_i) + '_hist_sel.png')
            plt.savefig(path)
            plt.clf()
            
#utils
def correct(A):
    for a in A:
        for (ind,aa) in enumerate(a):
            if aa is None:
                a[ind] = 0
            