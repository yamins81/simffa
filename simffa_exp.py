import numpy as np
import tabular as tb
import hyperopt
import simffa.simffa_experiments as sfexp
import simffa.simffa_bandit 
import simffa.simffa_feret

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

