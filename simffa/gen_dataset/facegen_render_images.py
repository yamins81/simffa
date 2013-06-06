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



# OBJECT_CATEGORIES = OrderedDict([
#   ('face_1', ['face1', 'face2', 'face3', 'face4', 'face5','face6',  'face7'])
# ])

OBJECT_CATEGORIES = OrderedDict([
  # ('face_1', ['fg_s001_0', 'fg_s001_2', 'fg_s001_4', 'fg_s001_6', 'fg_s001_8',
  #              'fg_s001_10', 'fg_s001_12', 'fg_s001_14']),
  # ('face_2', ['fg_s002_0', 'fg_s002_2', 'fg_s002_4', 'fg_s002_6', 'fg_s002_8',
  #              'fg_s002_10', 'fg_s002_12', 'fg_s002_14']),
  # ('face_3', ['fg_s003_0', 'fg_s003_2', 'fg_s003_4', 'fg_s003_6', 'fg_s003_8',
  #              'fg_s003_10', 'fg_s003_12', 'fg_s003_14']),
  # ('face_4', ['fg_s004_0', 'fg_s004_2', 'fg_s004_4', 'fg_s004_6', 'fg_s004_8',
  #              'fg_s004_10', 'fg_s004_12', 'fg_s004_14']),
  # ('face_5', ['fg_s005_0', 'fg_s005_2', 'fg_s005_4', 'fg_s005_6', 'fg_s005_8',
  #              'fg_s005_10', 'fg_s005_12', 'fg_s005_14']),
  # ('face_6', ['fg_s006_0', 'fg_s006_2', 'fg_s006_4', 'fg_s006_6', 'fg_s006_8',
  #              'fg_s006_10', 'fg_s006_12', 'fg_s006_14']),
  # ('face_7', ['fg_s007_0', 'fg_s007_2', 'fg_s007_4', 'fg_s007_6', 'fg_s007_8',
  #              'fg_s007_10', 'fg_s007_12', 'fg_s007_14']),
  # ('face_8', ['fg_s008_0', 'fg_s008_2', 'fg_s008_4', 'fg_s008_6', 'fg_s008_8',
  #              'fg_s008_10', 'fg_s008_12', 'fg_s008_14']),
  # ('face_9', ['fg_s009_0', 'fg_s009_2', 'fg_s009_4', 'fg_s009_6', 'fg_s009_8',
  #              'fg_s009_10', 'fg_s009_12', 'fg_s009_14']),
  # ('face_10', ['fg_s010_0', 'fg_s010_2', 'fg_s010_4', 'fg_s010_6', 'fg_s010_8',
  #              'fg_s010_10', 'fg_s010_12', 'fg_s010_14']),
  ('face_11', ['fg_s011_0', 'fg_s011_2', 'fg_s011_4', 'fg_s011_6', 'fg_s011_8',
               'fg_s011_10', 'fg_s011_12', 'fg_s011_14']),
  ('face_12', ['fg_s012_0', 'fg_s012_2', 'fg_s012_4', 'fg_s012_6', 'fg_s012_8',
               'fg_s012_10', 'fg_s012_12', 'fg_s012_14']),
  ('face_13', ['fg_s013_0', 'fg_s013_2', 'fg_s013_4', 'fg_s013_6', 'fg_s013_8',
               'fg_s013_10', 'fg_s013_12', 'fg_s013_14']),
  ('face_14', ['fg_s014_0', 'fg_s014_2', 'fg_s014_4', 'fg_s014_6', 'fg_s014_8',
               'fg_s014_10', 'fg_s014_12', 'fg_s014_14']),
  ('face_15', ['fg_s015_0', 'fg_s015_2', 'fg_s015_4', 'fg_s015_6', 'fg_s015_8',
               'fg_s015_10', 'fg_s015_12', 'fg_s015_14']),
  ('face_16', ['fg_s016_0', 'fg_s016_2', 'fg_s016_4', 'fg_s016_6', 'fg_s016_8',
               'fg_s016_10', 'fg_s016_12', 'fg_s016_14']),
  ('face_17', ['fg_s017_0', 'fg_s017_2', 'fg_s017_4', 'fg_s017_6', 'fg_s017_8',
               'fg_s017_10', 'fg_s017_12', 'fg_s017_14']),
  ('face_18', ['fg_s018_0', 'fg_s018_2', 'fg_s018_4', 'fg_s018_6', 'fg_s018_8',
               'fg_s018_10', 'fg_s018_12', 'fg_s018_14']),
  ('face_19', ['fg_s019_0', 'fg_s019_2', 'fg_s019_4', 'fg_s019_6', 'fg_s019_8',
               'fg_s019_10', 'fg_s019_12', 'fg_s019_14']),
  ('face_20', ['fg_s020_0', 'fg_s020_2', 'fg_s020_4', 'fg_s020_6', 'fg_s020_8',
               'fg_s020_10', 'fg_s020_12', 'fg_s020_14']),

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
            l['id'] =  model + str(_ind)#get_image_id(l)
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

    models = list(itertools.chain(*[v[::1] for v in OBJECT_CATEGORIES.values()]))
    model_categories = dict_inverse(OBJECT_CATEGORIES)
    
    templates = [{'n_objects': 1,
                  'n_ex': 2,
                  'name': 'OneObject', 
                  'template': {'bgname': choice(OBJECT_BACKGROUNDS),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1.,#uniform(0.8, 1.2),
                     'ty': 0,#uniform(-0.5, 0.5),
                     'tz': 0,#uniform(-0.5, 0.5),
                     'tx': 0,
                     'ryz': uniform(-10., 10.),
                     'rxy': uniform(-45., 45.),
                     'rxz': uniform(-10., 10.),
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
    print 'Making ' + str(imgs.shape[0]) + ' images'
    for i in xrange(imgs.shape[0]):
        print(meta['id'][i])
        # print imgs[i]
        curr_img_pre = imgs[i][::-1]

        curr_img = Image.fromarray(curr_img_pre).resize((512, 512), Image.ANTIALIAS)
        curr_img.save('%s/%s.png' % (dirname, meta['id'][i]))
        
    print str(imgs.shape[0]) + ' images done'


#main
dset = FaceGenDataset1
make_images(dset)