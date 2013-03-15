import numpy as np
import tabular as tb
import hyperopt
from hyperopt.mongoexp import MongoTrials
from hyperopt.utils import json_call

class InterleaveAlgo(hyperopt.BanditAlgo):
    def __init__(self, sub_algos, sub_exp_keys, priorities=None, **kwargs):
        if priorities is None:
            priorities = np.ones(len(sub_algos))
        else:
            priorities = np.array(priorities)
        assert (priorities >= 0).all()
        priorities = priorities.astype('float32') / priorities.sum()
        self.priorities = priorities
        
        hyperopt.BanditAlgo.__init__(self, sub_algos[0].bandit, **kwargs)
        # XXX: assert all bandits are the same
        self.sub_algos = sub_algos
        self.sub_exp_keys = sub_exp_keys
        if len(sub_algos) != len(sub_exp_keys):
            raise ValueError('algos and keys should have same len')
        # -- will be rebuilt naturally if experiment is continued
        self.stopped = set()

def suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, exp_keys, N):
    port = int(port)
    trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                        refresh=False)
    algos = []
    for bn, ban, ek in zip(bandit_names, bandit_algo_names, exp_keys):
        bandit = json_call(bn)
        subtrials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                         refresh=False, exp_key=ek)
        bak = {}
        bak['cmd'] = ('bandit_json evaluate', bn)
        algo = json_call(ban, (bandit,), bak)
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
	bandit_names = ['simffa_bandit.SimffaL1Bandit', 'simffa_bandit.SimffaL2Bandit', 'simffa_bandit.SimffaL3Bandit']
	N = 100
	# exp = suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, exp_keys, N)
	# return exp
	suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names, exp_keys, N)

startExp_main()

