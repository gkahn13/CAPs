import os
import argparse

from gcg.misc.utils import import_params
from gcg.algos.gcg_eval import GCGeval
from gcg.data.file_manager import CONFIGS_DIR

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('-itr', type=int)
parser.add_argument('-numrollouts', type=int)
args = parser.parse_args()

# load config
py_config_path = os.path.join(CONFIGS_DIR, '{0}.py'.format(args.exp))
assert(os.path.exists(py_config_path))
params = import_params(py_config_path)
with open(py_config_path, 'r') as f:
    params_txt = ''.join(f.readlines())

# create algorithm
AlgoClass = params['eval']['class']
assert(issubclass(AlgoClass, GCGeval))
algo = AlgoClass(eval_itr=args.itr,
                 num_rollouts=args.numrollouts,
                 eval_params=params['eval']['kwargs'],
                 exp_name=params['exp_name'],
                 env_eval_params=params['env_eval'] if params['env_eval'] else params['env'],
                 policy_params=params['policy'],
                 rp_eval_params=params['replay_pool_eval'],
                 seed=params['seed'],
                 log_level=params['log_level'])

# run algorithm
algo.run()
