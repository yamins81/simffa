import copy

from hyperopt.genson_helpers import (null,
                         false,
                         true,
                         choice,
                         uniform,
                         gaussian,
                         lognormal,
                         qlognormal,
                         ref)

lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : choice([.1,1,10]),
         'threshold' : choice([.1,1,10])
         }}

lpool = {'kwargs': {'stride' : 1,
          'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1,2,10])
         }}

rescale = {'kwargs': {'stride': 2}}

activ =  {'kwargs': {'min_out' : choice([null,0]),
                     'max_out' : choice([1,null])}}

filter1 = dict(
        initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16,32,64]),
            generate=(
                'random:uniform',
                {'rseed': choice([11, 12, 13, 14, 15])})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = choice([16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(10,15))})


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
                    filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
                    n_filters=choice([16,32,64]),
                    generate=('random:gabor',
                                     {'min_wl': 2, 'max_wl': 20 ,
                                      'rseed': choice([11, 12, 13, 14, 15])})
                                     ),
                kwargs={})

filter1_gabor_larger = dict(
                initialize=dict(
                    filter_shape=choice([(11,11),(15,15),(19,19),(23,23)]),
                    n_filters=choice([16,32,64]),
                    generate=('random:gabor',
                                     {'min_wl': 2, 'max_wl': 20 ,
                                      'rseed': choice([11, 12, 13, 14, 15])})
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




