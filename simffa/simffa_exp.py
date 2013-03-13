# import devthor.procedures as P
import numpy as np
import tabular as tb
import hyperopt
from hyperopt.mongoexp import MongoTrials
# from hyperopt.utils import json_call
# import devthor.new_new_bandits as nnb


def suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list):
    port = int(port)
    trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                        refresh=False)
    algos = []
    for bn, ban, ba, bk, baa, bak, ek in zip(bandit_names, bandit_algo_names, 
            bandit_args_list, bandit_kwargs_list, bandit_algo_args_list, bandit_algo_kwargs_list,
                                             exp_keys):
        bandit = json_call(bn, ba, bk)
        subtrials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                         refresh=False, exp_key=ek)
        if ba or bk:
            subtrials.attachments['bandit_data_' + ek] = cPickle.dumps((bn, ba, bk))
            bak['cmd'] = ('driver_attachment', 'bandit_data_' + ek)
        else:
            bak['cmd'] = ('bandit_json evaluate', bn)
        algo = json_call(ban, (bandit,) + baa, bak)
        algos.append(algo)
        
    algo = InterleaveAlgo(algos, exp_keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    if N is not None:
        exp.run(N, block_until_done=True)
    else:
        return exp


def startExp_main():
	dbname = 'simffa'
	exp_keys = ['SimffaL1BanditRandom', 'SimffaL2BanditRandom', 'SimffaL3BanditRandom']
	host = 'localhost'
	port = 22334
	bandit_algo_names = ['hyperopt.Random', 'hyperopt.Random', 'hyperopt.Random']
	bandit_names = ['SimffaL1Bandit', 'SimffaL2Bandit', 'SimffaL3Bandit']
	N = 50
	bandit_args_list = []
	bandit_kwargs_list = []
	bandit_algo_args_list = []
	bandit_algo_kwargs_list = []
	suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, 
                     exp_keys, N, bandit_args_list, bandit_kwargs_list, 
                     bandit_algo_args_list, bandit_algo_kwargs_list)
