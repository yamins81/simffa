import numpy as np
import tabular as tb
import hyperopt
import simffa.simffa_experiments as sfexp
import simffa.simffa_bandit 

def startExp_main():
	dbname = 'simffa_invar'
	exp_keys = ['SimffaL1BanditRandom', 'SimffaL2BanditRandom', 'SimffaL3BanditRandom']
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.Random'] * 3
	bandit_names = ['simffa.simffa_bandit.SimffaL1Bandit', 'simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [{} for _i in range(3)]
	bandit_kwargs_list = [{} for _i in range(3)]
	bandit_algo_args_list=[() for _i in range(3)]
	bandit_algo_kwargs_list=[{} for _i in range(3)]
	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	
	return exp


def simffaInvar():
	dbname = 'simffa_invar'
	# exp_keys = ['SimffaL1BanditRandom', 'SimffaL2BanditRandom', 'SimffaL3BanditRandom']
	exp_keys = ['SimffaL2BanditRandom', 'SimffaL3BanditRandom']
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.Random'] * 2
	# bandit_names = ['simffa.simffa_bandit.SimffaL1Bandit', 'simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_names = ['simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [{} for _i in range(2)]
	bandit_kwargs_list = [{} for _i in range(2)]
	bandit_algo_args_list=[() for _i in range(2)]
	bandit_algo_kwargs_list=[{} for _i in range(2)]
	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp

def simffaInvar_opt():
	dbname = 'simffa_invar'
	
	exp_keys = ['SimffaL2BanditParzen', 'SimffaL3BanditParzen']
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * 2
	
	bandit_names = ['simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [{} for _i in range(2)]
	bandit_kwargs_list = [{} for _i in range(2)]

	bandit_algo_args_list=[() for _i in range(2)]
	bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(2)]

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	
	return exp

def simffaInvar_opt_v1():
	dbname = 'simffa_invar_test'
	
	exp_keys = ['V1BanditParzen_eye']
	nExps = 1
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
	
	bandit_names = ['simffa.simffa_bandit.SimffaV1LikeBandit']
	bandit_args_list = [{} for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]

	bandit_algo_args_list=[() for _i in range(nExps)]
	bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	return exp

def simffaInvar_opt_shuffled():
	dbname = 'simffa_invar_test'
	
	exp_keys = ['V1BanditParzen_shuf','L2BanditParzen_shuf', 'L3BanditParzen_shuf']
	nExps = 3
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
	
	bandit_names = ['simffa.simffa_bandit.SimffaL1Bandit', 'simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [{1, True} for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]

	bandit_algo_args_list=[() for _i in range(nExps)]
	bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	
	return exp

def simffaInvar_opt_eye():
	dbname = 'simffa_invar_eye'
	
	exp_keys = ['V1BanditParzen_eye','L2BanditParzen_eye', 'L3BanditParzen_eye']
	nExps = 3
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
	
	bandit_names = ['simffa.simffa_bandit.SimffaV1LikeBandit', 'simffa.simffa_bandit.SimffaL2Bandit', 'simffa.simffa_bandit.SimffaL3Bandit']
	bandit_args_list = [{2} for _i in range(nExps)]
	bandit_kwargs_list = [{} for _i in range(nExps)]

	bandit_algo_args_list=[() for _i in range(nExps)]
	bandit_algo_kwargs_list=[{'gamma':0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

	N = None
	exp = sfexp.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
	
	return exp
