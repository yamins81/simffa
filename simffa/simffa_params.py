import copy

def string(s):
    return repr(s).replace("'",'"')

##Plain vanilla params  -- from FG11 paper

class Null(object):
    def __repr__(self):
        return 'null'
null = Null()

class FALSE(object):
    def __repr__(self):
        return 'false'
false = FALSE()

class TRUE(object):
    def __repr__(self):
        return 'true'
true = TRUE()


def repr(x):
    if isinstance(x,str):
        return '"' + str(x) + '"'
    else:
        return x.__repr__()

class gObj(object):
    def __init__(self,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        argstr = ', '.join([repr(x) for x in self.args])
        kwargstr = ', '.join([str(k) + '=' + repr(v) for k,v in self.kwargs.items()])

        astr = argstr + (', ' + kwargstr if kwargstr else '')
        return self.name + '(' + astr + ')'


class choice(gObj):
    name = 'choice'

class uniform(gObj):
    name = 'uniform'

class gaussian(gObj):
    name = 'gaussian'

class lognormal(gObj):
    name = 'lognormal'

class qlognormal(gObj):
    name = 'qlognormal'

class ref(object):
    def __init__(self,*p):
        self.path = p

    def __repr__(self):
        return '.'.join(self.path)


lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : choice([.1,1,10]),
         'threshold' : choice([.1,1,10])
         }}

lpool = {'kwargs': {'stride' : 2,
          'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1,2,10])
         }}


activ =  {'min_out' : choice([null,0]),
          'max_out' : choice([1,null])}

filter1 = dict(
        initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16,32,64]),
            generate=(
                'random:uniform',
                {'rseed': choice([11, 12, 13, 14, 15])})),
         kwargs=activ)

filter2 = dict(
        initialize=dict(
            filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
            n_filters=choice([16, 32, 64, 128]),
            generate=(
                'random:uniform',
                {'rseed': choice([21, 22, 23, 24, 25])})),
         kwargs=activ)

filter3 = dict(
        initialize=dict(
            filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
            n_filters=choice([16, 32, 64, 128, 256]),
            generate=(
                'random:uniform',
                {'rseed': choice([31, 32, 33, 34, 35])})),
         kwargs=activ)

layers = [[('lnorm', lnorm)],
          [('fbcorr', filter1),
           ('lpool', lpool),
           ('lnorm', lnorm)],
          [('fbcorr', filter2),
           ('lpool' , lpool),
           ('lnorm' , lnorm)],
          [('fbcorr', filter3),
           ('lpool', lpool),
           ('lnorm', lnorm)]
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
                kwargs=activ)

layers_gabor = [[('lnorm', lnorm)],
          [('fbcorr', filter1_gabor),
           ('lpool', lpool),
           ('lnorm', lnorm)],
          [('fbcorr', filter2),
           ('lpool' , lpool),
           ('lnorm' , lnorm)],
          [('fbcorr', filter3),
           ('lpool', lpool),
           ('lnorm', lnorm)]
         ]

l1_params_gabor = {'desc' : layers_gabor[:2]}
l2_params_gabor = {'desc' : layers_gabor[:3]}
l3_params_gabor = {'desc' : layers_gabor}




