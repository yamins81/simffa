import numpy as np
from scipy.stats import scoreatpercentile
import copy 
import hyperopt
from hyperopt.mongoexp import MongoTrials
import pymongo as pm

import simffa.simffa_experiments as sfexp
import simffa.simffa_params as sp 
# import simffa.simffa_feret
# import simffa.simffa_facegens


# optimization for behavioural tasks on facegen dataset
def simffa_facegen_task(dbname='simffa_facegen', randomSearch=False):
	
	host = 'localhost'
	port = 22334
	nExps = 3 

	bandit_names = ['simffa.simffa_bandit_facegens.Simffa_FaceGenTasks_Bandit_V1',
					'simffa.simffa_bandit_facegens.Simffa_FaceGenTasks_Bandit_L2',
					'simffa.simffa_bandit_facegens.Simffa_FaceGenTasks_Bandit_L3']
	bandit_args_list = [() for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]
	exp_keys = ['simffa_facegen_V1', 'simffa_facegen_L2', 'simffa_facegen_L3']

	if randomSearch:
		bandit_algo_names = ['hyperopt.Random'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{} for _i in range(nExps)]
	else:
		bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp

# re-extract top models (based on tasks) and run neural fits
def simffa_facegen_reextract_neural(dbname_old, dbname_new):
	
	host = 'localhost'
	port = 22334
	conn = pm.Connection(host, port)   
	
	new_bandit = 'simffa.simffa_bandit_issa.Simffa_IssaNeural_Bandit'
	exp_keys = ['simffa_facegen_V1', 'simffa_facegen_L2', 'simffa_facegen_L3']

	for key in exp_keys:
		jobs_old = conn[dbname_old]['jobs'].find({'exp_key':key,'state':2})
		new_trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname_new),
		             refresh=False, exp_key=key)
		new_ids = new_trials.new_trial_ids(1)

		acc = np.array([x['result']['results']['id_accuracy'] for x in jobs_old])
		acc_thres = scoreatpercentile(acc[~np.isnan(acc)],99)
		t = np.array(np.where(acc >= acc_thres))
		t = np.unique(t)

		job_old = conn[dbname_old]['jobs'].find({'exp_key':key, 'state':2})[t[0]]
		new_id = new_ids[0]
		
		new_trials.insert_trial_docs(sfexp.reExtract(job_old, new_bandit, new_id))
		new_trials.refresh()
		sfexp.block_until_done(new_trials, 2)

	return

# re-extract top models (based on tasks) and run fsi analyses
def simffa_facegen_reextract_fsi(dbname_old, dbname_new):
	
	host = 'localhost'
	port = 22334
	conn = pm.Connection(host, port)   
	
	new_bandit = 'simffa.simffa_bandit_fbo.Simffa_FboFSI_Bandit'
	exp_keys = ['simffa_facegen_V1', 'simffa_facegen_L2', 'simffa_facegen_L3']

	for key in exp_keys:
		jobs_old = conn[dbname_old]['jobs'].find({'exp_key':key,'state':2})
		new_trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname_new),
		             refresh=False, exp_key=key)
		new_ids = new_trials.new_trial_ids(1)

		acc = np.array([x['result']['results']['id_accuracy'] for x in jobs_old])
		acc_thres = scoreatpercentile(acc[~np.isnan(acc)],99)
		t = np.array(np.where(acc >= acc_thres))
		t = np.unique(t)

		job_old = conn[dbname_old]['jobs'].find({'exp_key':key, 'state':2})[t[0]]
		new_id = new_ids[0]
		
		new_trials.insert_trial_docs(sfexp.reExtract(job_old, new_bandit, new_id))
		new_trials.refresh()
		sfexp.block_until_done(new_trials, 2)

	return

