import numpy as np
import copy
from pyll import scope
from hyperopt.pyll_utils import hp_uniform, hp_loguniform, hp_quniform, hp_qloguniform
from hyperopt.pyll_utils import hp_normal, hp_lognormal, hp_qnormal, hp_qlognormal
from hyperopt.pyll_utils import hp_choice

lnis = hp_choice('lnis', [(3,3),(5,5),(7,7),(9,9)])
lnorm = {'kwargs': {'inker_shape' : lnis,
        'outker_shape' : lnis,
        'remove_mean' : hp_choice('lnrm', [0, 1]),
        'stretch' : hp_choice('lns', [.1,1,10]),
        'threshold' : hp_choice('lnt',[.1,1,10])
         }}

lpool = {'kwargs': {'stride' : 1,
        'ker_shape' : hp_choice('lpks', [(3,3),(5,5),(7,7),(9,9)]),
        'order' : hp_choice('lpo', [1,2,10])
         }}

rescale = {'kwargs': {'stride': 2}}

activ =  {'kwargs': {'min_out' : hp_choice('amin1', 
                             [None, hp_uniform('aminval1', -1, .5)]),
                      'max_out' : hp_choice('amax1', 
                             [hp_uniform('amaxval1', .8, 1.5), None])}}

# activ =  {'kwargs': {'min_out' : hp_choice('amin1', [null,0]),
                      # 'max_out' : hp_choice('amax1', [1,null]) }}

filter1 = dict(
        initialize=dict(
            filter_shape=hp_choice('filter_shape', [(3,3),(5,5),(7,7),(9,9)]),
            n_filters=hp_choice('n_filters', [16,32,64]),
            generate=(
                'random:uniform',
                {'rseed': hp_choice('rseed',[11, 12, 13, 14, 15])})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = hp_choice('filter2',[16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': hp_choice('rseed', range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = hp_choice('filter3', [16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': hp_choice('rseed', range(10,15))})


layers = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
           ]

l1_params = {'desc' : layers[:2]}
l2_params = {'desc' : layers[:3]}
l3_params = {'desc' : layers}

filter1_gabor = dict(
                initialize=dict(
                    filter_shape=hp_choice('filter_shape',[(3,3),(5,5),(7,7),(9,9)]),
                    n_filters=hp_choice('n_filters',[16,32,64]),
                    generate=('random:gabor',
                                     {'min_wl': 2, 'max_wl': 20 ,
                                      'rseed': hp_choice('rseed',[11, 12, 13, 14, 15])})
                                     ),
                kwargs={})

filter1_gabor_larger = dict(
                initialize=dict(
                    filter_shape=hp_choice('filter_shape',[(11,11),(15,15),(19,19),(23,23)]),
                    n_filters=hp_choice('n_filters',[16,32,64]),
                    generate=('random:gabor',
                                     {'min_wl': 2, 'max_wl': 20 ,
                                      'rseed': hp_choice('rseed', [11, 12, 13, 14, 15])})
                                     ),
                kwargs={})

layers_gabor = [[('lnorm', lnorm)],
            [('fbcorr', filter1_gabor),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
           ]

layers_gabor_larger = [[('lnorm', lnorm)],
            [('fbcorr', filter1_gabor_larger),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            ]


l1_params_gabor = {'desc' : layers_gabor[:2]}
l2_params_gabor = {'desc' : layers_gabor[:3]}
l3_params_gabor = {'desc' : layers_gabor}

l1_params_gabor_larger = {'desc' : layers_gabor_larger[:2]}
l2_params_gabor_larger = {'desc' : layers_gabor_larger[:3]}
l3_params_gabor_larger = {'desc' : layers_gabor_larger}



######v1

v1like_lpool = {'kwargs': {'ker_shape' : hp_choice('v1ks', [(13,13),(17,17),(19,19),(21,21)]),
          'order' : hp_choice('v1o', [1, 2, 10])
         }}

v1like_filter = dict(
         initialize=dict(
            filter_shape=hp_choice('filter_shape',[(33,33),(43,43),(53,53)]),
            n_filters=hp_choice('n_filters', [16,64,96]),
            generate=(
                'random:gabor',
                {'min_wl': 2, 'max_wl': 20 ,
                 'rseed': hp_choice('rseed', range(5))})),
         kwargs={})

v1like_rescale = {'kwargs': {'stride' : hp_choice('stride', [3, 4, 5])}}

v1d = [[('lnorm', lnorm),
        ('fbcorr', v1like_filter),
        ('lnorm', lnorm),
        ('activ', activ),
        ('lpool', v1like_lpool),
        ('rescale', v1like_rescale)]]
             
v1like_params = {'desc': v1d}


######pixels
pixels_lpool = {'kwargs': {'ker_shape' : hp_choice('ker_shape', [(1,1),(5,5)]),
                           'order' : 1}
               }
pixels_rescale = {'kwargs': {'stride' : hp_choice('stride', [1, 3])}}

pixels_params = {'desc': [
                [('lpool', pixels_lpool),
                 ('rescale', pixels_rescale)],
             ]}
             
             
######v1like subsets

v1_fap = [[('fbcorr', v1like_filter),
           ('activ', activ),
           ('lpool', v1like_lpool),
           ('rescale', v1like_rescale)]]
           
v1_fp = [[('fbcorr', v1like_filter),
           ('lpool', v1like_lpool),
           ('rescale', v1like_rescale)]]

v1like_spectrum_params = {'desc': hp_choice('desc', [v1_fap, v1_fp] + 
                                         [[[v1d[0][_i] for _i in [i, -1]]] for i in range(1,len(v1d[0])-1)] + \
                                         [[[v1d[0][_i] for _i in range(i) + [-1] ]] for i in range(1,len(v1d[0]))]
                          )}



