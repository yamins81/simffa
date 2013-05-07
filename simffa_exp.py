import numpy as np
import tabular as tb
from scipy.stats import scoreatpercentile

import hyperopt
from hyperopt.mongoexp import MongoTrials
import pymongo as pm

import simffa.simffa_experiments as sfexp
import simffa.simffa_bandit 
import simffa.simffa_feret
import simffa.simffa_facegens

def simffa_exp_main(dbname='simffa', randomSearch=False, label_set=1, shuf=False):
	host = 'localhost'
	port = 22334
	
	bandit_names = ['simffa.simffa_bandit.SimffaV1LikeBandit', 'simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [(label_set, shuf) for _i in range(3)]
	bandit_kwargs_list = [{} for _i in range(3)]

	if randomSearch:
		bandit_algo_names = ['hyperopt.Random'] * 3
		bandit_algo_args_list=[() for _i in range(3)]
		bandit_algo_kwargs_list=[{} for _i in range(3)]
	else:
		bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * 3
		bandit_algo_args_list=[() for _i in range(3)]
		bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(3)]

	dataset_tag = 'Face'
	if label_set == 2:
		dataset_tag = 'Eye'
	suffix = '_'
	if shuf:
		suffix = '_shuf_'
	algo_tag = 'Parzen'
	if randomSearch:
		algo_tag = 'Random'
	prefix = 'Simffa' + algo_tag + dataset_tag + suffix
	exp_keys = [prefix+'V1', prefix+'L2', prefix+'L3']

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp



def simffa_exp_pruned(dbname='simffa_feret', randomSearch=False):
	host = 'localhost'
	port = 22334
	nExps = 2 

	bandit_names = ['simffa.simffa_feret.FERETL3Bandit', 'simffa.simffa_feret.FERETL2Bandit']
	bandit_args_list = [() for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]

	if randomSearch:
		bandit_algo_names = ['hyperopt.Random'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{} for _i in range(nExps)]
	else:
		bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	exp_keys = ['simffa_feret_L3', 'simffa_feret_L2']

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp



def simffa_exp_pruned_facegen(dbname='simffa_facegen', randomSearch=False):
	host = 'localhost'
	port = 22334
	nExps = 2 

	bandit_names = ['simffa.simffa_facegens.Simffa_FaceGen_L3Bandit', 'simffa.simffa_facegens.Simffa_FaceGen_L2Bandit']
	bandit_args_list = [() for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]

	if randomSearch:
		bandit_algo_names = ['hyperopt.Random'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{} for _i in range(nExps)]
	else:
		bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
		bandit_algo_args_list=[() for _i in range(nExps)]
		bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	exp_keys = ['simffa_facegen_L3', 'simffa_facegen_L2']

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp



def simffa_exp_repruned_facegen():
	dbname_old = 'simffa_facegen'
	dbname_new ='simffa_facegen_selected'# + str(selection_criterion_index)
	host = 'localhost'
	port = 22334

	exp_keys = ['simffa_facegen_L3', 'simffa_facegen_L2']
	selection_criterions = ['avg_pose_accuracy', 'id_accuracy', 'express_accuracy']

	conn = pm.Connection(host, port)    
	new_trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname_new),
	                 refresh=False, exp_key=exp_keys)
	
	for key in exp_keys:
		jobs_old = conn[dbname_old]['jobs'].find({'exp_key':key,'state':2})
		t = []
		# pick out top 10% of models in each behavioural task
		for s_c in selection_criterions:
			print 'for task ' + s_c
			acc = np.array([x['result']['results'][s_c] for x in jobs_old])
			acc_ = acc[~np.isnan(acc)]
			if acc_.shape[0] == 0:
				continue
			acc_thres = scoreatpercentile(acc[~np.isnan(acc)],90)
			t_ = np.array(np.where(acc >= acc_thres))
			t.append(t_)
		t = np.unique(np.array(t))
		# test them on neural fitting
		for i in range(len(t)):
			print 're-extracting ' + str(len(t)) + key + ' models  '
			job_old = conn[dbname_old]['jobs'].find({'exp_key':key, 'state':2})[t[i]]
			results_old = job_old['result']['results']
			new_id = new_trials.new_trial_ids(1)
			new_misc = job_old['misc']
			new_misc['tid'] = new_id
			new_misc['task_stats'] = results_old
			new_misc['cmd'][1] = new_misc['cmd'][1] + '_v2'
			new_result = {'status': hyperopt.STATUS_NEW}
			new_docs = new_trials.new_trial_docs([new_id],
			            [None], [new_result], [new_misc])
			new_trials.insert_trial_docs(new_docs)
			new_trials.refresh()

	sfexp.block_until_done(new_trials, 2)
	return
