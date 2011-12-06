import numpy as np
import scipy as sp


#############
#####sgd#####
#############
import asgd  # use master branch from https://github.com/jaberg/asgd
from early_stopping import fit_w_early_stopping, EarlyStopping


def train_asgd_classifier_normalize(train_Xy, test_Xy, verbose=False, batchsize=10,
        step_sizes=(3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6), trace_normalize=False):

    trainXy, test_Xy, train_mean, train_std, trace = normalize([train_Xy, test_Xy],
                                                                     trace_normalize=trace_normalize)
    model, es, result = train_asgd_classifier(train_Xy, test_Xy, verbose=verbose, batchsize=batchsize,
                                       step_sizes=step_sizes)

    result['train_mean'] = train_mean
    result['train_std'] = train_std
    result['trace'] = trace
    return model, es, result


def train_asgd_classifier(train_Xy, test_Xy,
        verbose=False,
        batchsize=10,
        validation_interval=100,
        step_sizes=(3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)):
    """
    batchsize = 10                 # unit: examples
    validation_interval = 100      # unit: batches
    """
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy
    if train_X.ndim != 2: raise TypeError()
    if test_X.ndim != 2: raise TypeError()
    n_examples, n_features = train_X.shape
    labelset = set(train_y)
    labelset.update(test_y)

    if verbose:
        print 'training classifier: n_train=%i n_test=%i n_features=%i' % (
                n_examples, len(test_X), n_features)

    assert labelset == set(range(len(labelset)))

    if labelset == set([0, 1]):
        labels = [1, 1]
        # BINARY CLASSIFICATION
        # -- change labels to -1, +1
        train_y = train_y * 2 - 1
        test_y = test_y * 2 - 1
        def model_fn(step_size0):
            return asgd.naive_asgd.NaiveBinaryASGD(
                    n_features=n_features,
                    l2_regularization=1e-3,
                    sgd_step_size0=step_size0)

        if use_theano:
            print "train_classifier not using theano for binary classif."
    else:
        # MULTI-CLASS CLASSIFICATION
        labels = range(len(labelset))
        def model_fn(step_size0):
            return asgd.naive_asgd.NaiveMulticlassASGD(
                    n_features=n_features,
                    n_classes=len(labelset),
                    l2_regularization=1e-3,
                    sgd_step_size0=step_size0)

    # -- repeat training for several learning rates
    #    take model that was best on held-out data
    results = [fit_w_early_stopping(
                model_fn(step_size0),
                es=EarlyStopping(warmup=50, max_iters=1000), # unit: validation intervals
                train_X=train_X,
                train_y=train_y,
                validation_X=test_X,
                validation_y=test_y,
                batchsize=batchsize,
                validation_interval=validation_interval,
                verbose=verbose
                )
            for step_size0 in step_sizes]

    results.sort(cmp=lambda a, b: cmp(a[1].best_y, b[1].best_y))
    model, es, test_prediction, train_prediction = results[0]
    result = get_result(train_y, test_y, train_prediction, test_prediction, labels)
    return model, es, result


#############
##liblinear##
#############
from scikits.learn import svm as sklearn_svm
from scikits.learn import linear_model as sklearn_linear_model
from scikits.learn.linear_model.logistic import LogisticRegression


def train_liblinear_classifier(train_Xy,
                               test_Xy,
                               classifier_kwargs,
                               relabel = True,
                               trace_normalize=False):

    """
    Classifier using the built-in multi-class classification capabilities of liblinear
    """

    train_features, train_labels = train_Xy,
    test_features, test_labels = test_Xy

    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    if relabel:
        label_to_id = dict([(k,v) for v, k in enumerate(labels)])
        train_ids = sp.array([label_to_id[i] for i in train_labels])
    else:
        train_ids = train_labels
    train_Xz = (train_features, train_ids)
    classifier, train_mean, train_std, trace = train_liblinear_classifier_core(train_Xz
                                                                             trace_normalize=trace_normalize,
                                                                             **classifier_kwargs)
    train_data = {'train_mean':train_mean, 'train_std': train_std, 'trace': trace}
    (test_Xy, train_mean, train_std, trace) = normalize([test_Xy],
                                                                             data=train_data,
                                                                             trace_normalize=trace_normalize)
    (test_features, test_labels) = test_Xy

    test_prediction = labels[classifier.predict(test_features)]
    train_prediction = labels[classifier.predict(train_features)]
    result = get_result(train_labels, test_labels, train_prediction, test_prediction, labels)
    result['cls_data']['train_mean'] = train_mean
    result['cls_data']['train_std'] = test_std
    result['cls_data']['trace'] = trace
    return classifier, None, result


def train_liblinear_classifier_core(trainXy,
                     classifier_type = "liblinear",
                     trace_normalize=False,
                     **kwargs
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    classifier_type = liblinear or libsvm"""

    #do normalization
    (train_features, train_labels), train_mean, train_std, trace = normalize([trainXy],
                                                                              trace_normalize=trace_normalize)
    if classifier_type == 'liblinear':
        clf = sklearn_svm.LinearSVC(**kwargs)
    if classifier_type == 'libSVM':
        clf = sklearn_svm.SVC(**kwargs)
    elif classifier_type == 'LRL':
        clf = LogisticRegression(**kwargs)
    elif classifier_type == 'MCC':
        clf = CorrelationClassifier(**kwargs)
    elif classifier_type.startswith('svm.'):
        ct = classifier_type.split('.')[-1]
        clf = getattr(sklearn_svm,ct)(**kwargs)
    elif classifier_type.startswith('linear_model.'):
        ct = classifier_type.split('.')[-1]
        clf = getattr(sklearn_linear_model,ct)(**kwargs)

    clf.fit(train_features, train_labels)

    return clf, train_mean, train_std, trace



############
#evaluation#
############

def evaluate_classifier_normalize(model, test_Xy, data, trace_normalize=False, verbose=False, batchsize=10):
    test_Xy = normalize([test_Xy], data=data, trace_normalize=trace_normalize)
    return evaluate_classifier(model, (test_X, test_y), batchsize=batchsize, verbose=verbose)


def evaluate_classifier(model, test_Xy, labels,
        batchsize=10,
        verbose=0):

    test_X, test_y = test_Xy

    if set(test_y) == set([0, 1]):
        test_y = test_y * 2 - 1
    assert set(test_y) == set(labels)

    tpos = 0
    vpos = 0
    errs = []
    test_prediction = []
    while vpos < len(test_X):
        xi = test_X[vpos:vpos + batchsize]
        yi = test_y[vpos:vpos + batchsize]
        pi = model.predict(xi)
        test_prediction.extend(pi.tolist())
        assert np.all(np.isfinite(pi))
        errs.append((yi != pi).astype('float64'))
        vpos += batchsize
    test_prediction = np.array(test_prediction)

    vscore = np.mean(errs)
    # -- std dev appropriate for classification
    vscore_std = np.sqrt(vscore * (1.0 - vscore) / len(test_X))
    result = get_test_result(test_y, test_prediction, labels)
    return result



#########
##stats##
#########


def get_result(train_labels, test_labels, train_prediction, test_prediction, labels):
    result = {'train_errors': (train_labels != train_prediction).tolist(),
     'test_errors': (test_labels != test_prediction).tolist(),
     'train_prediction': train_prediction.tolist(),
     'test_prediction' : test_prediction.tolist(),
     'labels' : labels,
     }
    stats = multiclass_stats(test_labels, test_prediction, train_labels, train_prediction, labels)
    result.update(stats)
    return result


def get_test_result(test_labels, test_prediction, labels):
    result = {
     'test_errors': (test_labels != test_prediction).tolist(),
     'test_prediction' : test_prediction.tolist(),
     'labels': labels
     }
    stats = multiclass_test_stats(test_labels, test_prediction, labels)
    result.update(stats)
    return result


def multiclass_stats(test_actual, test_predicted, train_actual, train_predicted,labels):
    test_results = multiclass_test_stats(test_actual, test_predicted, labels)
    train_results = multiclass_test_stats(train_actual, train_predicted, labels, prefix='train')
    test_results.update(train_results)
    return test_results


def multiclass_test_stats(test_actual, test_predicted, labels, prefix='test'):
    test_accuracy = float(100*(test_prediction == test_labels).sum() / float(len(test_prediction)))
    test_aps = []
    test_aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        test_prec,test_rec = precision_and_recall(test_actual,test_predicted,label)
        test_ap = ap_from_prec_and_rec(test_prec,test_rec)
        test_aps.append(test_ap)
        test_auc = auc_from_prec_and_rec(test_prec,test_rec)
        test_aucs.append(test_auc)
    test_ap = np.array(test_aps).mean()
    test_auc = np.array(test_aucs).mean()
    return {prefix+'accuracy' : test_accuracy,
            prefix+'ap' : test_ap,
            prefix+'auc' : test_auc}


def precision_and_recall(actual,predicted,cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp /sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec,rec


def ap_from_prec_and_rec(prec,rec):
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec,rec):
    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc




#########
##utils##
#########

def normalize(feats_Xy, trace_normalize=True, data=None):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """
    feats, labels = zip(*feats_Xy)
    if data is not None:
        train_f = feats[0]
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
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
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats_Xy = tuple(zip(feats,labels))
    return feats_Xy + (m, s, tr)


def mean_and_std(X, min_std):
    # -- this loop is more memory efficient than numpy
    #    but not as numerically accurate as possible
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


