import copy
import os
import itertools

import numpy as np
import genthor.datasets as gd
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
  ('objectome_models', ['weimaraner',
 'lo_poly_animal_TRTL_B',
 'lo_poly_animal_ELE_AS1',
 'lo_poly_animal_TRANTULA',
 'foreign_cat',
 'lo_poly_animal_CHICKDEE',
 'lo_poly_animal_HRS_ARBN',
 'MB29346',
 'MB31620',
 'MB29874',
 'interior_details_033_2',
 'MB29822',
 'face7',
 'single_pineapple',
 'pumpkin_3',
 'Hanger_02',
 'MB31188',
 'antique_furniture_item_18',
 'MB27346',
 'interior_details_047_1',
 'laptop01',
 'womens_stockings_01M',
 'pear_obj_2',
 'household_aid_29',
 '22_acoustic_guitar',
 'MB30850',
 'MB30798',
 'MB31015',
 'Nurse_pose01',
 'fast_food_23_1',
 'kitchen_equipment_knife2',
 'flarenut_spanner',
 'womens_halterneck_06',
 'dromedary',
 'MB30758',
 'MB30071',
 'leaves16',
 'lo_poly_animal_DUCK',
 '31_african_drums',
 'lo_poly_animal_RHINO_2',
 'lo_poly_animal_ANT_RED',
 'interior_details_103_2',
 'interior_details_103_4',
 'MB27780',
 'MB27585',
 'build51',
 'Colored_shirt_03M',
 'calc01',
 'Doctor_pose02',
 'bullfrog',
 'MB28699',
 'jewelry_29',
 'trousers_03',
 '04_piano',
 'womens_shorts_01M',
 'womens_Skirt_02M',
 'lo_poly_animal_TIGER_B',
 'MB31405',
 'MB30203',
 'zebra',
 'lo_poly_animal_BEAR_BLK',
 'lo_poly_animal_RB_TROUT',
 'interior_details_130_2',
 'Tie_06'])
])

OBJECT_BACKGROUNDS = ['graybg.jpg']

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
            l['id'] =  model + str(_ind)
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
                                 
class MonkeyObjectomeDataset(gd.GenerativeBase):
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

# no background
class MonkeyObjectomeDataset_v0(MonkeyObjectomeDataset):   

    home = '/mindhive/dicarlolab/u/esolomon/.skdata/'
    BASE_NAME = 'genthor'
    GENTHOR_PATH = os.path.join(home, BASE_NAME)
    RESOURCE_PATH = os.path.join(GENTHOR_PATH, "resources")
    CACHE_PATH = os.path.join(GENTHOR_PATH, "cache")
    BACKGROUND_PATH = os.path.join(RESOURCE_PATH, "backgrounds")
    OBJ_PATH = os.path.join(RESOURCE_PATH, "objs")
    EGG_PATH = os.path.join(RESOURCE_PATH, "eggs")
    BAM_PATH = os.path.join(RESOURCE_PATH, "bams")
    TEXTURE_PATH = os.path.join(RESOURCE_PATH, "textures")

    models = list(itertools.chain(*[v[::1] for v in OBJECT_CATEGORIES.values()]))
    model_categories = dict_inverse(OBJECT_CATEGORIES)
    
    templates = [{'n_objects': 1,
                  'n_ex': 100,
                  'name': 'OneObject', 
                  'template': {'bgname': choice(['graybg.jpg']),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(2./3), np.log(2.)),
                     'tx': uniform(-1.0, 1.0),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                    
                     }
                  }]   
               

def make_images(dsetclass):
    import os
    import cPickle
    import Image

    dirname = '/mindhive/dicarlolab/u/esolomon/rishi_imgs/obj_M/'
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
        curr_img_pre = imgs[i][::-1]
        curr_img = Image.fromarray(curr_img_pre).resize((512, 512), Image.ANTIALIAS)
        curr_img.save('%s/%s.png' % (dirname, meta['id'][i]))
        
    print str(imgs.shape[0]) + ' images done'

# main
dset = MonkeyObjectomeDataset_v0
make_images(dset)