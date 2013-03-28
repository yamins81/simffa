import copy

import numpy as np
import scipy as sp
try:
    import mcc
except ImportError:
    print("Can't import separate mcc package")

from sklearn.metrics.pairwise import pairwise_distances 
import scipy.stats as st

#############
#####asgd#####
#############
try:
    import asgd  # use master branch from https://github.com/jaberg/asgd
except ImportError:
    print("Can't import asgd.")

def train_asgd(train_Xy,
               test_Xy,
               normalization=True,
               trace_normalize=False):
               
    
    model, train_data = train_only_asgd(train_Xy,
                                        normalization=normalization,
                                        trace_normalize=trace_normalize)
                                        
    model, train_result = evaluate(model,
            train_Xy,
            train_data,
            regression=False,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='train')
            
    model, test_result = evaluate(model,
            test_Xy,
            train_data,
            regression=False,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='test')
    train_result.update(test_result)
            
    return model, train_result


def train_only_asgd(train_Xy, 
                    normalization=True,
                    trace_normalize=False,
                    margin_biases=None):
    """
    """
    train_features, train_labels = train_Xy
    if train_features.ndim != 2: raise TypeError()
    n_examples, n_features = train_features.shape
    labelset = set(train_labels)
    
    #do normalization
    if normalization:
        train_features, train_mean, train_std, trace = normalize(
                              [train_features], trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None
                                
    if labelset == set([-1, 1]):
        labelset = np.array([-1, 1])
        model = asgd.naive_asgd.NaiveBinaryASGD(n_features=n_features)
    else:
        # MULTI-CLASS CLASSIFICATION
        assert len(labelset) > 2
        assert set(labelset) == set(range(len(labelset)))
        labelset = np.arange(len(labelset))
        model = asgd.naive_asgd.NaiveOVAASGD(n_features=n_features,
                                                    n_classes=len(labelset))
                              
    
    model.fit(train_features, train_labels, margin_biases=margin_biases)
    train_data = {'train_mean': train_mean,
                  'train_std': train_std,
                  'trace': trace,
                  'labelset': labelset,
                  'labelmap': None}

    return model, train_data



#############
###scikits###
#############
try:
    from sklearn import svm as sklearn_svm
    from sklearn import linear_model as sklearn_linear_model
except ImportError:
    print("Can't import scikits stuff")

def train_scikits(train_Xy,
                  test_Xy,
                  model_type,
                  regression=False,
                  model_kwargs=None,
                  fit_kwargs=None,
                  normalization=True,
                  trace_normalize=False,
                  margins=False):

    model, train_data = train_only_scikits(train_Xy,
                  model_type,
                  regression=regression,
                  model_kwargs=model_kwargs,
                  fit_kwargs=fit_kwargs,
                  normalization=normalization,
                  trace_normalize=trace_normalize)

    model, train_result = evaluate(model,
            train_Xy,
            train_data,
            regression=regression,
            normalization=normalization,
            trace_normalize=trace_normalize,                                
            prefix='train',
            margins=margins)
            
    model, test_result = evaluate(model,
            test_Xy,
            train_data,
            regression=regression,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='test',
            margins=margins)
    train_result.update(test_result)
            
    return model, train_result


def train_only_scikits(train_Xy,
                  model_type,
                  regression=False,
                  model_kwargs=None,
                  fit_kwargs=None,
                  normalization=True,
                  trace_normalize=False):

    """

    """

    train_features, train_labels = train_Xy

    if not regression:
        labelset = sp.unique(train_labels)
        label_to_id = dict([(k,v) for v, k in enumerate(labelset)])
        train_ids = sp.array([label_to_id[i] for i in train_labels])
    else:
        train_ids = train_labels
        labelset = None
        
    labelmap = labelset

    #do normalization
    if normalization:
        train_features, train_mean, train_std, trace = normalize(
                              [train_features], trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None
    model = train_scikits_core(train_features, train_ids, model_type, model_kwargs,
                              fit_kwargs, labelset)
    train_data = {'train_mean':train_mean,
                  'train_std': train_std,
                  'trace': trace,
                  'labelset': labelset,
                  'labelmap': labelmap}

    return model, train_data


def train_scikits_core(train_features,
                     train_labels,
                     model_type,
                     model_kwargs,
                     fit_kwargs,
                     labelset
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    model_type = liblinear or libsvm"""
    if model_kwargs is None:
        model_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    if model_type == 'liblinear':
        cls = sklearn_svm.LinearSVC
    elif model_type == 'libSVM':
        cls = sklearn_svm.SVC
    elif model_type == 'LRL':
        cls = sklearn_linear_model.LogisticRegression
    elif model_type == 'MCC':
        cls = MaximumCorrelationClassifier
        model_kwargs['labelset'] = np.arange(len(labelset))
        model_kwargs['n_features'] = train_features.shape[1]
    elif model_type == 'MCC2':
        cls = MaximumCorrelationClassifier2
        model_kwargs['labelset'] = np.arange(len(labelset))
        model_kwargs['n_features'] = train_features.shape[1]
    elif model_type == 'MCC3':
        cls = MaximumCorrelationClassifier3
        model_kwargs['labelset'] = np.arange(len(labelset))
        model_kwargs['n_features'] = train_features.shape[1]
    elif model_type == 'MCC_nico':
        cls = mcc.MaximumCorrelationClassifier
        model_kwargs['n_features'] = train_features.shape[1]
    elif model_type.startswith('svm.'):
        ct = model_type.split('.')[-1]
        cls = getattr(sklearn_svm,ct)
    elif model_type.startswith('linear_model.'):
        ct = model_type.split('.')[-1]
        cls = getattr(sklearn_linear_model,ct)
    else:
        raise ValueError('Model type %s not recognized' % model_type)

    clf = cls(**model_kwargs)
    clf.fit(train_features, train_labels, **fit_kwargs)
    ### DEBUG {{{
    ### print '***', sorted(clf.coef_[0])
    ### print '!!!', len(train_features)
    ### print '!!!', train_labels[0]
    ### print '!!!', train_features[0]
    ### }}}
    return clf

#############
#correlation#
#############


class MaximumCorrelationClassifier(object):
    """Implemntation from 
       http://www.comp.lancs.ac.uk/~kristof/research/notes/min_dist_class/index.html
    """
    def __init__(self, labelset, n_features, fnorm=False, snorm=False): 
        assert labelset.ndim == 1
        assert len(labelset) == len(np.unique(labelset))
        self.labelset = labelset
        self.n_classes = len(labelset)
        self.n_features = n_features
        self.fnorm = fnorm
        self.snorm = snorm

    def initialize(self, val=None, n=None):
        self._mu = np.zeros((self.n_classes, self.n_features))
        self._n_samples = np.zeros((self.n_classes,)).astype(np.int)
        if val is not None:
            assert val.shape == self.mu.shape
            assert n.dtype == self._n_samples.dtype
            assert n.shape == self._n_samples.shape
            assert (n >= 0).all()
            self._mu += val
            self._n_samples += n
            
    def fit(self, X, y):
        self.initialize(val=None, n=None)
        if self.fnorm:
            X = X - X.mean(1)[:,np.newaxis]
            X = X / np.sqrt((X**2).sum(1))[:, np.newaxis]
        self.partial_fit(X, y)
    
    def partial_fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape == (len(y), self.n_features)
        assert set(y) <= set(self.labelset), (set(y), set(self.labelset))
        uy = np.unique(y)
        for vi, v in enumerate(uy):
            Xv = X[y == v]
            Xvm = Xv.mean(0)
            ns = self._n_samples[vi]
            self._mu[vi] = (ns / (ns + 1.)) * self._mu[vi] + (1. / (ns + 1.)) * Xvm
            self._n_samples[vi] += 1
            
    @property
    def coef_(self):
        c = self._mu
        if self.snorm:
            c = c - c.mean(1)[:, np.newaxis]
            c = c / np.sqrt((c**2).sum(1))[:, np.newaxis]
        return c
    
    @property
    def intercept_(self):
        c = self._mu
        if self.snorm:
            c = c - c.mean(1)[:, np.newaxis]
            c = c / np.sqrt((c**2).sum(1))[:, np.newaxis]
        return -(1./2) * (c**2).sum(1)
    
    def decision_function(self, X):
        if self.fnorm:
            X = X - X.mean(1)[:,np.newaxis]
            X = X / np.sqrt((X**2).sum(1))[:, np.newaxis]
        return np.dot(X, self.coef_.T) + self.intercept_
    
    def predict(self, X, bias=None):
        assert X.ndim == 2
        assert X.shape[1] == self.n_features
        decs = self.decision_function(X)
        if bias is not None:
            decs += bias
        return self.labelset[decs.argmax(1)]
        
        
class MaximumCorrelationClassifier2(object):
    """
    """
    def __init__(self, labelset, n_features, fnorm=False, snorm=False): 
        assert labelset.ndim == 1
        assert len(labelset) == len(np.unique(labelset))
        self.labelset = labelset
        self.n_classes = len(labelset)
        self.n_features = n_features
        self.fnorm = fnorm
        self.snorm = snorm

    def initialize(self, val=None, n=None):
        self._mu = np.zeros((self.n_classes, self.n_features))
        self._n_samples = np.zeros((self.n_classes,)).astype(np.int)
        if val is not None:
            assert val.shape == self.mu.shape
            assert n.dtype == self._n_samples.dtype
            assert n.shape == self._n_samples.shape
            assert (n >= 0).all()
            self._mu += val
            self._n_samples += n
            
    def fit(self, X, y):
        self.initialize(val=None, n=None)
        if self.fnorm:
            X = X - X.mean(1)[:,np.newaxis]
            X = X / np.sqrt((X**2).sum(1))[:, np.newaxis]
        self.partial_fit(X, y)
    
    def partial_fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape == (len(y), self.n_features)
        assert set(y) <= set(self.labelset), (set(y), set(self.labelset))
        uy = np.unique(y)
        assert (uy == self.labelset).all()
        for vi, v in enumerate(self.labelset):
            Xv = X[y == v]
            Xvm = Xv.mean(0)
            ns = self._n_samples[vi]
            nv = float(Xv.shape[0])
            self._mu[vi] = (ns / (ns + nv)) * self._mu[vi] + (nv / (ns + nv)) * Xvm
            self._n_samples[vi] += int(nv)
            
    @property
    def coef_(self):
        c = self._mu
        if self.snorm:
            c = c - c.mean(1)[:, np.newaxis]
            c = c / np.sqrt((c**2).sum(1))[:, np.newaxis]
        return c
        
    def decision_function(self, X):
        if self.fnorm:
            X = X - X.mean(1)[:,np.newaxis]
            X = X / np.sqrt((X**2).sum(1))[:, np.newaxis]

        # this is faster than yamutils.stats.pearsonr
        return 1 - pairwise_distances(X, self.coef_, metric='correlation')
    
    def predict(self, X, bias=None):
        assert X.ndim == 2
        assert X.shape[1] == self.n_features
        decs = self.decision_function(X)
        if bias is not None:
            decs += bias
        return self.labelset[decs.argmax(1)]


class MaximumCorrelationClassifier3(object):
    """From Ha's implementation
    """
    def __init__(self, labelset, n_features, fnorm=None, snorm=None): 
        """fnorm and snorm are silently ignored"""
        assert labelset.ndim == 1
        assert len(labelset) == len(np.unique(labelset))
        self.labelset = np.array(labelset)
        self.n_classes = len(labelset)
        self.n_features = n_features

    def initialize(self):
        self._mu = np.zeros((self.n_classes, self.n_features))
            
    def fit(self, X, y):
        self.initialize()
        # -- self.partial_fit(X, y)
        for ic, c in enumerate(self.labelset):
            self._mu[ic] = X[y == c].mean(0)
            
    @property
    def coef_(self):
        c = self._mu
        return c
        
    def decision_function(self, X):
        #return ystats.pearsonr(X.T, self.coef_. T)[0]
        c = self.coef_
        return np.array([[st.pearsonr(x, c0)[0] for c0 in c] for x in X])
    
    def predict(self, X):
        assert X.ndim == 2
        assert X.shape[1] == self.n_features
        decs = self.decision_function(X)
        return self.labelset[decs.argmax(1)]


############
#evaluation#
############

def evaluate(model,
            test_Xy,
            train_data,
            regression=False,
            normalization=True,
            trace_normalize=False,
            prefix='test',
            batchsize=None,
            bias=None,
            margins=False):

    if len(test_Xy[1]) == 0:
        return model, copy.deepcopy(train_data)
    
    test_features, test_labels = test_Xy
    if normalization:
        test_features, train_mean, train_std, trace = normalize(
              [test_features], data=train_data, trace_normalize=trace_normalize)
    
    if batchsize is None:
        test_prediction = model.predict(test_features)
    else:
        test_prediction = batch_prediction(model, test_features, batchsize)

    if regression:
        result = regression_stats(test_labels, test_prediction, prefix=prefix)
    else:
        test_prediction = test_prediction.astype(np.int)
        labelset = train_data['labelset']
        labelmap = train_data['labelmap']
        if labelmap is not None:
            test_prediction = labelmap[test_prediction]
        result = get_test_result(test_labels, test_prediction, labelset, prefix=prefix, bias=bias)
    if margins:
        result[prefix + '_margins'] = model.decision_function(test_features)
    result.update(train_data)
    return model, result


def batch_prediction(model, test_X, batchsize):
    pos = 0
    test_prediction = []
    while pos < len(test_X):
        xi = test_X[pos:pos + batchsize]
        pi = model.predict(xi)
        test_prediction.extend(pi.tolist())
        assert np.all(np.isfinite(pi))
        pos += batchsize
    return np.array(test_prediction)



#########
##stats##
#########
from scipy.stats.stats import pearsonr

def get_regression_result(train_actual, test_actual, train_predicted, test_predicted):
    test_results = regression_stats(test_actual, test_predicted, prefix='test')
    train_results = regression_stats(train_actual, train_predicted, prefix='train')
    test_results.update(train_results)
    return test_results


def regression_stats(actual, predicted, prefix='test'):
    return {prefix+'_rsquared' : rsquared(actual, predicted),
            prefix+'_mean_error' : mean_error(actual, predicted),
            prefix + '_prediction': predicted.tolist()}


def get_result(train_labels, test_labels, train_prediction, test_prediction, labelset):
    result = {'train_errors': (train_labels != train_prediction).tolist(),
     'test_errors': (test_labels != test_prediction).tolist(),
     'train_prediction': train_prediction.tolist(),
     'test_prediction' : test_prediction.tolist(),
     'labelset' : labelset,
     }
    stats = multiclass_stats(test_labels, test_prediction, train_labels, train_prediction, labelset)
    result.update(stats)
    return result


def get_test_result(test_labels, test_prediction, labelset, prefix='test', bias=None):
    result = {
     prefix + '_errors': (test_labels != test_prediction).tolist(),
     prefix + '_prediction' : test_prediction.tolist(),
     'labelset': labelset
     }
    stats = multiclass_test_stats(test_labels, test_prediction, labelset, prefix=prefix, bias=bias)
    result.update(stats)
    return result


def multiclass_stats(test_actual, test_predicted, train_actual, train_predicted, labelset, bias=None):
    test_results = multiclass_test_stats(test_actual, test_predicted, labelset, prefix='test', bias=bias)
    train_results = multiclass_test_stats(train_actual, train_predicted, labelset, prefix='train')
    test_results.update(train_results)
    return test_results


def multiclass_test_stats(test_actual, test_predicted, labelset, prefix='test', bias=None):
    accuracy, ap, auc = classification_stats(test_actual, test_predicted, labelset, bias=bias)    
    confusion_matrix = get_confusion_matrix(test_actual, test_predicted, labelset, bias=bias)
    return {prefix+'_accuracy' : accuracy,
            prefix+'_ap' : ap,
            prefix+'_auc' : auc,
            prefix+'_cm': confusion_matrix.tolist()}


def get_confusion_matrix(actual, predicted, labelset, bias=None):
    if bias is None:
        bias = np.ones((len(predicted),)).astype(np.int)
    else:   
        check_bias(bias, predicted)
        bias = bias * len(predicted)
    f = lambda l1, l2: np.dot(((actual == l1) & (predicted == l2)), bias)
    return np.array([[f(label1, label2) for label1 in labelset] for label2 in labelset])


def check_bias(bias, predicted):
    assert (bias >= 0).all(), (bias.shape, bias.min())
    assert (1 - bias.sum() < .001), bias.sum()
    assert bias.shape == predicted.shape, (bias.shape, predicted.shape)
    
    
def classification_stats(actual, predicted, labelset, bias=None):
    if bias is None:
        bias = np.ones((len(predicted), )).astype(np.float) / len(predicted)
    check_bias(bias, predicted)
    accuracy = np.dot(100*(predicted == actual), bias)
    aps = []
    aucs = []
    if len(labelset) == 2:
        labelset = labelset[1:]
    for label in labelset:
        prec, rec = precision_and_recall(actual, predicted, label, bias=bias)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    ap = np.array(aps).mean()
    auc = np.array(aucs).mean()
    return accuracy, ap, auc
    

def average_precision(actual, predicted, labelset):
    if len(labelset) == 2:
        labelset = labelset[1:]
    aps = []
    for label in labelset:
        prec, rec = precision_and_recall(actual, predicted, label)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
    ap = np.array(aps).mean()
    return ap
    

def ap_from_prec_and_rec(prec, rec):
    ap = 0
    rng = np.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def area_under_curve(actual, predicted, labelset):
    if len(labelset) == 2:
        labelset = labelset[1:]
    aucs = []
    for label in labelset:
        prec, rec = precision_and_recall(actual, predicted, labelset)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    auc = np.array(aucs).mean()
    return auc
    

def auc_from_prec_and_rec(prec, rec):
    #area under curve
    h = np.diff(rec)
    auc = np.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc

## this definition of rsquared gives negative values -rishi
# def rsquared(actual, predicted):
#     a_mean = actual.mean()
#     num = np.linalg.norm(actual - predicted) ** 2
#     denom = np.linalg.norm(actual - a_mean) ** 2
#     return 1 -  num / denom

def rsquared(actual, predicted):
    r,p = pearsonr(actual, predicted)
    rsq = r**2
    return rsq
    
def mean_error(actual, predicted):
    num = np.linalg.norm(actual - predicted) ** 2
    return num 


def precision_and_recall(actual, predicted, cls, bias=None):
    if bias is None:
        bias = np.ones((len(predicted),)).astype(float) / len(predicted)
    check_bias(bias, predicted)
    bias  = bias * len(predicted)
    c = (actual == cls) * bias
    si = np.argsort(-c)
    tp = np.cumsum(np.single(predicted[si] == cls) * bias)
    fp = np.cumsum(np.single(predicted[si] != cls) * bias)
    rec = tp / np.dot(predicted == cls, bias)
    prec = tp / (fp + tp)
    return prec, rec



#########
##utils##
#########

def normalize(feats, trace_normalize=False, data=None):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """

    if data is None:
        train_f = feats[0]
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
        # Ha's implementation {{{
        # s = train_f.std(axis=0)
        # s[np.abs(s) < 1e-6] = 1
        # }}}
    else:
        m = data['train_mean']
        s = data['train_std']
    feats = [(f - m) / s for f in feats]
    if trace_normalize:
        if data is None:
            train_f = feats[0]
            tr = np.maximum(np.sqrt((train_f**2).sum(axis=1)).mean(), 1e-8)
        else:
            tr = data['trace']
    else:
        tr = None
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats = tuple(feats)
    return feats + (m, s, tr)


def mean_and_std(X, min_std):
    # XXX: this loop is more memory efficient than numpy but not as
    # numerically accurate. It would be better to look at the train_mean,
    # and then either use the msq for getting unit norms if the train_means
    # are small-ish, or else use numpy.std if the mean is large enough to
    # cause numerical trouble
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-4,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    train_mean, train_std = mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


def simple_bracket_min(f, pt0, pt1):
    v0 = f(pt0)
    v1 = f(pt1)
    if v0 > v1:
        while v0 > v1:
            raise NotImplementedError()

