import copy
import os
import itertools

import numpy as np
# import genthor.datasets as gd
import facegen_dataset as gd    
import genthor.model_info as gmi

import tabular as tb 

import pyll
choice = pyll.scope.choice
uniform = pyll.scope.uniform
loguniform = pyll.scope.loguniform
import pyll.stochastic as stochastic

from yamutils.basic import dict_inverse

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
    except ImportError:
        raise ImportError("OrderedDict not available")

OBJECT_CATEGORIES = OrderedDict([
  ('face_1', ['fg_s001_0', 'fg_s001_2', 'fg_s001_4', 'fg_s001_6', 'fg_s001_8',
               'fg_s001_10', 'fg_s001_12', 'fg_s001_14']),
  ('face_2', ['fg_s002_0', 'fg_s002_2', 'fg_s002_4', 'fg_s002_6', 'fg_s002_8',
               'fg_s002_10', 'fg_s002_12', 'fg_s002_14']),
  ('face_3', ['fg_s003_0', 'fg_s003_2', 'fg_s003_4', 'fg_s003_6', 'fg_s003_8',
               'fg_s003_10', 'fg_s003_12', 'fg_s003_14']),
  ('face_4', ['fg_s004_0', 'fg_s004_2', 'fg_s004_4', 'fg_s004_6', 'fg_s004_8',
               'fg_s004_10', 'fg_s004_12', 'fg_s004_14']),
  ('face_5', ['fg_s005_0', 'fg_s005_2', 'fg_s005_4', 'fg_s005_6', 'fg_s005_8',
               'fg_s005_10', 'fg_s005_12', 'fg_s005_14']),
])
OBJECT_BACKGROUNDS = ['whitebg.jpg']

get_image_id = gd.get_image_id

def get_oneobj_latents(tdict, models, categories):
    rng = np.random.RandomState(seed=0)
    latents = []
    tname = tdict['name']
    template = tdict['template']
    n_ex = tdict['n_ex']
    for model in models:
        # print('Generating meta for %s' % model)
        for _ind in range(n_ex):
            l = stochastic.sample(template, rng)
            l['obj'] = [model]
            l['category'] = [categories[model][0]]
            l['id'] = model + '_' + str(np.random.randint(1000))#get_image_id(l)
            rec = (l['bgname'],
                   float(l['bgphi']),
                   float(l['bgpsi']),
                   float(l['bgscale']),
                   l['category'],
                   l['obj'],
                   [float(l['ryz'])],
                   [float(l['rxz'])],
                   [float(l['rxy'])],
                   [float(l['ty'])],
                   [float(l['tz'])],
                   [float(l['tx'])],
                   [float(l['s'])],
                   [None],
                   [None],                   
                   tname,
                   l['id'])
            latents.append(rec)
    return latents
                                 
class FaceGenDataset(gd.GenerativeBase):
    def _get_meta(self):
        models = self.models
        templates = self.templates
        use_canonical = self.use_canonical
        internal_canonical = self.internal_canonical
        
        latents = []
        rng = np.random.RandomState(seed=0)
        model_categories = self.model_categories
    
        for tdict in templates:
            latents.extend(get_oneobj_latents(tdict, models, model_categories))
            
        ids = [_x[-1] for _x in latents]
        print len(ids)
        print len(set(ids))
        assert len(ids) == len(set(ids))
        idlen = max(map(len, ids))
        tnames = [_x[-2] for _x in latents]
        tnamelen = max(map(len, tnames))

        meta = tb.tabarray(records=latents, names = ['bgname',
                                                     'bgphi',
                                                     'bgpsi',
                                                     'bgscale',
                                                     'category',
                                                     'obj',
                                                     'ryz',
                                                     'rxz',
                                                     'rxy',
                                                     'ty',
                                                     'tz',
                                                     'tx',
                                                     's',
                                                     'texture', 
                                                     'texture_mode',
                                                     'tname',
                                                     'id'], 
                           formats=['|S20', 'float', 'float', 'float'] + \
                                  ['|O8']*11 +  ['|S%s' % tnamelen, '|S%s' % idlen])
            
    
        if internal_canonical:
            meta = meta.addcols([np.ones((len(meta),))], names = ['internal_canonical'])
        n_objs = map(len, meta['obj'])
        meta = meta.addcols([n_objs], names = ['n_objects'])
                
        return meta            

class FaceGenDataset1(FaceGenDataset):   

    home = '/mindhive/dicarlolab/u/esolomon/.skdata/'
    BASE_NAME = 'genthor'
    GENTHOR_PATH = os.path.join(home, BASE_NAME)
    RESOURCE_PATH = os.path.join(GENTHOR_PATH, "resources")
    CACHE_PATH = os.path.join(GENTHOR_PATH, "cache")
    BACKGROUND_PATH = os.path.join(RESOURCE_PATH, "backgrounds")
    OBJ_PATH = os.path.join(RESOURCE_PATH, "objs")
    # RESOURCE_PATH = '/mindhive/dicarlolab/u/rishir/FaceGen_Small/'
    EGG_PATH = os.path.join(RESOURCE_PATH, "eggs")
    BAM_PATH = os.path.join(RESOURCE_PATH, "bams")
    TEXTURE_PATH = os.path.join(RESOURCE_PATH, "textures")

    models = list(itertools.chain(*[v[::2] for v in OBJECT_CATEGORIES.values()]))
    model_categories = dict_inverse(OBJECT_CATEGORIES)
    
    templates = [{'n_objects': 40,
                  'n_ex': 5,
                  'name': 'OneObject', 
                  'template': {'bgname': choice(OBJECT_BACKGROUNDS),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(0.8, 1.2),
                     'ty': uniform(-0.5, 0.5),
                     'tz': uniform(-0.5, 0.5),
                     'tx': 0,
                     'ryz': uniform(-20., 20.),
                     'rxy': uniform(-90., 90.),
                     'rxz': uniform(-20., 20.),
                     }
                  }]   
                  
import os
import cPickle
import Image
def make_images(dsetclass):
    dirname = '/mindhive/dicarlolab/u/esolomon/rishi_imgs/'
    
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    dataset = dsetclass(internal_canonical=True)
    meta = dataset.meta
    lspec1 = [{'name': 'alight1', 'type': 'AmbientLight', 'color':(2.0,2.0,2.0,1)},
              {'name': 'alight2', 'type': 'PointLight', 'color':(2.5, 2.5, 2.5,1), 'pos': (50, -50, 50)}]
    lspecs = [lspec1 for _ind in range(len(meta))]
    names = meta.dtype.names + ('light_spec',) 
    formats = zip(*meta.dtype.descr)[1] + ('|O8',)

    meta= tb.tabarray(records=[tuple(meta[i]) + (lspecs[i],) for i in range(len(meta))], names=names, formats=formats)
    imgs = dataset.get_images({'dtype':'uint8', 'size':(1024, 1024), 'normalize':False, 'mode':'L'},
                              global_light_spec=None, get_models=True)

    with open (os.path.join(dirname, 'metadata.pkl'), 'w') as _f:
        cPickle.dump(meta, _f)  
    for i in xrange(imgs.shape[0]):
        Image.fromarray(imgs[i][::-1]).resize((512, 512), Image.ANTIALIAS).save('%s/%s.png' % (dirname, meta['id'][i]))
        print(meta['id'][i])


#main
dset = FaceGenDataset1
make_images(dset)