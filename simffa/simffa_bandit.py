from thoreano import TheanoSLM
import skdata.fbo
import hyperopt.genson_bandits as gb

import simffa_params
from classifier import (train_classifier_normalize,
                        evaluate_classifier_normalize,
                        train_classifier,
                        evaluate_classifier)
from theano_slm import slm_from_config, FeatureExtractor



class SimffaL3Bandit(gb.GensonBandit):
    source_string = cvpr_params.string(simffa_params.l3_params)

    def __init__(self):
        super(SimffaL3Bandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl):

        dataset = skdata.fbo.FaceBodyObject20110803()
        X, y = dataset.image_classification_task()
        slm = slm_from_config(config, X.shape, batchsize=4, use_theano=True)
        extractor = FeatureExtractor(X, slm, batchsize=batchsize)
        features = extractor.compute_features(use_memmap=False)




        return result
