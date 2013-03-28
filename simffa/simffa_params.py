import numpy as np

from pyll import scope
from hyperopt.pyll_utils import hp_uniform, hp_loguniform, hp_quniform, hp_qloguniform
from hyperopt.pyll_utils import hp_normal, hp_lognormal, hp_qnormal, hp_qlognormal
from hyperopt.pyll_utils import hp_choice

nsc0 =  scope.int(hp_quniform('nsc0',2,16,1))
nsc1 =  scope.int(hp_quniform('nsc1',2,16,1))
nsc1r = scope.int(hp_quniform('nsc1f',2,5,1))
nsc2 =  scope.int(hp_quniform('nsc2',2,16,1))
nsc3 =  scope.int(hp_quniform('nsc3',2,16,1))
nsc4 =  scope.int(hp_quniform('nsc4',2,10,1))

lnorm0 = {'kwargs':{'inker_shape' : (nsc0, nsc0),
         'outker_shape' : (nsc0,nsc0),
         'remove_mean' : hp_choice('lnrm0',[0,1]),
         'stretch' : hp_uniform('lns0',0,50),
         'threshold' : hp_choice('lnt0',[None, hp_uniform('lnta0',0,50)])
         }}
lnorm1 = {'kwargs':{'inker_shape' : (nsc1,nsc1),
         'outker_shape' : (nsc1,nsc1),
         'remove_mean' : hp_choice('lnrm1',[0,1]),
         'stretch' : hp_uniform('lns1',0,50),
         'threshold' : hp_choice('lnt1',[None, hp_uniform('lnta1',0,50)])
         }}
lnorm1r = {'kwargs':{'inker_shape' : (nsc1r,nsc1r),
         'outker_shape' : (nsc1r,nsc1r),
         'remove_mean' : hp_choice('lnrm1',[0,1]),
         'stretch' : hp_uniform('lns1',0,50),
         'threshold' : hp_choice('lnt1',[None, hp_uniform('lnta1',0,50)])
         }}
lnorm2 = {'kwargs':{'inker_shape' : (nsc2,nsc2),
          'outker_shape' : (nsc2,nsc2),
         'remove_mean' : hp_choice('lnrm2',[0,1]),
         'stretch' : hp_uniform('lns2',0,50),
         'threshold' : hp_choice('lnt2',[None, hp_uniform('lnta2',0,50)])
         }}
lnorm3 = {'kwargs':{'inker_shape' : (nsc3,nsc3),
         'outker_shape' : (nsc3,nsc3),
         'remove_mean' : hp_choice('lnrm3',[0,1]),
         'stretch' : hp_uniform('lns3',0,50),
         'threshold' : hp_choice('lnt3',[None, hp_uniform('lnta3',0,50)])
         }}
lnorm4 = {'kwargs':{'inker_shape' : (nsc4,nsc4),
         'outker_shape' : (nsc4,nsc4),
         'remove_mean' : hp_choice('lnrm4',[0,1]),
         'stretch' : hp_uniform('lns4',0,50),
         'threshold' : hp_choice('lnt4',[None, hp_uniform('lnta4',0,50)])
         }}

rescale1 = {'kwargs': {'stride': 2}}
rescale2 = {'kwargs': {'stride': 2}}
rescale3 = {'kwargs': {'stride': 2}}
rescale4 = {'kwargs': {'stride': 2}}
#rescale1 = {'kwargs': {'stride': hp_choice('stride1',[1,2])}}

lp1 = scope.int(hp_quniform('lp1',1,16,1))
lp1r = scope.int(hp_quniform('lp1r',1,3,1))
lp2 = scope.int(hp_quniform('lp2',1,16,1))
lp3 = scope.int(hp_quniform('lp3',1,16,1))
lp4 = scope.int(hp_quniform('lp4',1,10,1))
lpool1 = {'kwargs': {'ker_shape' : (lp1,lp1),
          'order' : hp_loguniform('lpo1',np.log(1), np.log(50))
         }}
lpool1r = {'kwargs': {'ker_shape' : (lp1r,lp1r),
          'order' : hp_loguniform('lpo1',np.log(1), np.log(50))
         }}
lpool2 = {'kwargs': {'ker_shape' : (lp2,lp2),
                     'order' : hp_loguniform('lpo2',np.log(1), np.log(50))
         }}
lpool3 = {'kwargs': {'ker_shape' : (lp3,lp3),
          'order' : hp_loguniform('lpo3',np.log(1), np.log(50))
         }}
lpool4 = {'kwargs': {'ker_shape' : (lp4,lp4),
          'order' : hp_loguniform('lpo4',np.log(1), np.log(50))
         }}
         
activ1 =  {'kwargs': {'min_out' : hp_choice('amin1',[None, hp_uniform('aminval1',-0.3,.2)]),
                     'max_out' : hp_choice('amax1',[hp_uniform('amaxval1',.8,1.3), None])}}
activ2 =  {'kwargs': {'min_out' : hp_choice('amin2',[None, hp_uniform('aminval2',-0.3,.2)]),
                      'max_out' : hp_choice('amax2',[hp_uniform('amaxval2',.8,1.3), None])}}
activ3 =  {'kwargs': {'min_out' : hp_choice('amin3',[None, hp_uniform('aminval3',-0.3,.2)]),
                      'max_out' : hp_choice('amax3',[hp_uniform('amaxval3',.8,1.3), None])}}
activ4 =  {'kwargs': {'min_out' : hp_choice('amin4',[None, hp_uniform('aminval4',-0.3,.2)]),
                      'max_out' : hp_choice('amax4',[hp_uniform('amaxval4',.8,1.3), None])}}                 

fs1 = scope.int(hp_quniform('fs1',2,12,1))
fs1r = scope.int(hp_quniform('fs1r',2,5,1))
fs2 = scope.int(hp_quniform('fs2',2,12,1))
fs3 = scope.int(hp_quniform('fs3',2,12,1))
fs4 = scope.int(hp_quniform('fs4',2,12,1))
filter1 = dict(
         initialize=dict(
            filter_shape=(fs1,fs1),
            n_filters=scope.int(hp_qloguniform('fsn1',np.log(16), np.log(32),q=4)),
            generate=hp_choice('l1fchoice',[('random:gabor',
                {'min_wl': 2, 'max_wl': 20 ,
                 'rseed': hp_choice('fse1_gabor',range(5))}) , 
                 ('random:uniform',
                {'rseed': hp_choice('fse1_uniform', range(5)),
                 'fmean': hp_uniform('fmean1',-0.2,.2),
                 'fnorm': hp_uniform('fnorm1',.8,1.2)})])),
         kwargs={})


filter1r = dict(
         initialize=dict(
            filter_shape=(fs1r,fs1r),
            n_filters=scope.int(hp_qloguniform('fsn1',np.log(16), np.log(32),q=4)),
            generate= ('random:uniform',
                {'rseed': hp_choice('fse1_uniform', range(5)),
                 'fmean': hp_uniform('fmean1',-0.2,.2),
                 'fnorm': hp_uniform('fnorm1',.8,1.2)})),
         kwargs={})


filter2 = dict(
         initialize=dict(
            filter_shape=(fs2,fs2),
            n_filters=scope.int(hp_qloguniform('fsn2',np.log(16),np.log(48),q=4)),
            generate=(
                'random:uniform',
                {'rseed': hp_choice('fse2', range(5,10)),
                 'fmean': hp_uniform('fmean2',-0.2,.2),
                 'fnorm': hp_uniform('fnorm2',.8,1.2)})),
         kwargs={})
         
filter3 = dict(
         initialize=dict(
            filter_shape=(fs3,fs3),
            n_filters=96,
            generate=(
                'random:uniform',
                {'rseed': hp_choice('fse3', range(10,15)),
                 'fmean': hp_uniform('fmean3',-0.2,.2),
                 'fnorm': hp_uniform('fnorm3',.8,1.2)})),
         kwargs={})

filter4 = dict(
         initialize=dict(
            filter_shape=(fs3,fs3),
            n_filters=scope.int(hp_qloguniform('fsn4',np.log(48),np.log(96),q=4)),
            generate=(
                'random:uniform',
                {'rseed': hp_choice('fse4', range(15,20)),
                 'fmean': hp_uniform('fmean4',-0.2,.2),
                 'fnorm': hp_uniform('fnorm4',.8,1.2)})),
         kwargs={})


l1_params = {'desc': [
            [('fbcorr', filter1r),
             ('activ', activ1),
             ('lpool', lpool1r),
             ('rescale', rescale1),
             ('lnorm', lnorm1r)],
           ]}

#rishi edit
# l1_params = {'desc': [
#             [('fbcorr', filter1r),
#              ('activ', activ1),
#              ('lpool', lpool1r),
#              ('lnorm', lnorm1r)],
#            ]}


l2_params = {'desc': [[('lnorm', lnorm0)],
            [('fbcorr', filter1),
             ('activ', activ1),
             ('lpool', lpool1),
             ('rescale', rescale1),
             ('lnorm', lnorm1)],
            [('fbcorr', filter2),
             ('activ', activ2),
             ('lpool', lpool2),
             ('rescale', rescale2),
             ('lnorm', lnorm2)]
           ]}

l3_params = {'desc': [[('lnorm', lnorm0)],
            [('fbcorr', filter1),
             ('activ', activ1),
             ('lpool', lpool1),
             ('rescale', rescale1),
             ('lnorm', lnorm1)],
            [('fbcorr', filter2),
             ('activ', activ2),
             ('lpool', lpool2),
             ('rescale', rescale2),
             ('lnorm', lnorm2)],
            [('fbcorr', filter3),
             ('activ', activ3),
             ('lpool', lpool3),
             ('rescale', rescale3),
             ('lnorm', lnorm3)],
           ]}
           
l4_params = {'desc': [[('lnorm', lnorm0)],
            [('fbcorr', filter1),
             ('activ', activ1),
             ('lpool', lpool1),
             ('rescale', rescale1),
             ('lnorm', lnorm1)],
            [('fbcorr', filter2),
             ('activ', activ2),
             ('lpool', lpool2),
             ('rescale', rescale2),
             ('lnorm', lnorm2)],
            [('fbcorr', filter3),
             ('activ', activ3),
             ('lpool', lpool3),
             ('rescale', rescale3),
             ('lnorm', lnorm3)],
            [('fbcorr', filter4),
             ('activ', activ4),
             ('lpool', lpool4),
             ('rescale', rescale4),
             ('lnorm', lnorm4)],             
           ]}
           
v1lps = scope.int(hp_quniform('v1lps', 2,25,1))
v1like_lpool = {'kwargs': {'ker_shape' : (v1lps, v1lps),
          'order' : hp_loguniform('v1lpo',np.log(1), np.log(10))
         }}

v1fn1=scope.int(hp_qloguniform('v1fn1',np.log(16),np.log(64),q=4))
v1like_filter = dict(
         initialize=dict(
            filter_shape=(v1fn1, v1fn1),
            n_filters=96,
            generate=(
                'random:gabor',
                {'min_wl': 2, 'max_wl': 20 ,
                 'rseed': hp_choice('v1frs',range(5))})),
         kwargs={})
         
lnorm_in = {'kwargs':{'inker_shape' : (nsc1,nsc1),
         'outker_shape' : (nsc1,nsc1),
         'remove_mean' : hp_choice('lnrm_in',[0,1]),
         'stretch' : hp_uniform('lns_in',0,50),
         'threshold' : hp_choice('lnt_in',[None, hp_uniform('lnta_in',0,50)])
         }}

lnorm_out = {'kwargs':{'inker_shape' : (nsc2,nsc2),
         'outker_shape' : (nsc2,nsc2),
         'remove_mean' : hp_choice('lnrm_out',[0,1]),
         'stretch' : hp_uniform('lns_out',0,50),
         'threshold' : hp_choice('lnt_out',[None, hp_uniform('lnta_out',0,50)])
         }}

v1like_rescale = {'kwargs': {'stride' : scope.int(hp_quniform('v1rs',3, 4, 5))}}

v1like_params = {'desc': [
            [('lnorm', lnorm_in),
             ('fbcorr', v1like_filter),
             ('lnorm', lnorm_out),
             ('activ', activ1),
             ('lpool', v1like_lpool),
             ('rescale', v1like_rescale)]]}