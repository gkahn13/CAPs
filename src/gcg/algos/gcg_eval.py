import os, sys, glob
import subprocess

from gcg.samplers.sampler import Sampler
from gcg.data.logger import logger
from gcg.data import mypickle
from gcg.misc import utils
from gcg.data.file_manager import FileManager
from gcg.replay_pools.replay_pool import ReplayPool


class GCGeval(object):

    def __init__(self,
                 eval_itr,
                 num_rollouts,
                 eval_params,
                 exp_name,
                 env_eval_params,
                 policy_params,
                 rp_eval_params,
                 seed=None, log_level='info',  log_fname='log_eval.txt'):
        self._eval_itr = eval_itr
        self._num_rollouts = num_rollouts

        ### create file manager and setup logger
        self._fm = FileManager(exp_name,  is_continue=True, log_level=log_level, log_fname=log_fname,
                               log_folder='eval_itr_{0:04d}'.format(self._eval_itr))

        logger.debug('Git current')
        logger.debug(subprocess.check_output('git status | head -n 1', shell=True).decode('utf-8').strip())
        logger.debug(subprocess.check_output('git log -n 1| head -n 1', shell=True).decode('utf-8').strip())

        logger.debug('Seed {0}'.format(seed))
        utils.set_seed(seed)

        ### create environments
        self._env_eval = env_eval_params['class'](params=env_eval_params['kwargs'])

        ### create policy
        self._policy = policy_params['class'](
            env_spec=self._env_eval.spec,
            exploration_strategies=[],
            inference_only=True,
            **policy_params['kwargs']
        )

        ### create replay pools
        self._save_async = True
        self._replay_pool_eval = ReplayPool(env_spec=self._env_eval.spec,
                                            obs_history_len=self._policy.obs_history_len,
                                            N=self._policy.N,
                                            labeller=None,
                                            size=int(5 * self._env_eval.horizon),
                                            save_rollouts=True,
                                            save_rollouts_observations=True,
                                            save_env_infos=True)

        ### create samplers
        self._sampler_eval = Sampler(
            env=self._env_eval,
            policy=self._policy,
            replay_pool=self._replay_pool_eval
        )

    #############
    ### Files ###
    #############

    @property
    def _save_dir(self):
        eval_save_dir = os.path.join(self._fm.exp_dir, 'eval_itr_{0:04d}'.format(self._eval_itr))
        os.makedirs(eval_save_dir, exist_ok=True)
        return eval_save_dir

    def _rollouts_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_eval_rollouts.pkl'.format(itr))

    ############
    ### Save ###
    ############

    def _save_rollouts(self, itr, rollouts):
        fname = self._rollouts_file_name(itr)
        mypickle.dump({'rollouts': rollouts}, fname, async=self._save_async)

    ###############
    ### Restore ###
    ###############

    def _load_rollouts(self, itr):
        fname = self._rollouts_file_name(itr)
        if os.path.exists(fname):
            rollouts = mypickle.load(fname)['rollouts']
        else:
            rollouts = []
        return rollouts

    ############
    ### Eval ###
    ############

    def _reset(self, **kwargs):
        self._sampler_eval.reset(**kwargs)

    def _step(self):
        self._sampler_eval.step(step=0,
                                take_random_actions=False,
                                explore=False)

    def _log(self):
        self._env_eval.log()
        self._replay_pool_eval.log()
        self._policy.log()
        logger.dump_tabular(print_func=logger.info)

    def _save(self, rollouts, new_rollouts):
        assert (len(new_rollouts) > 0)

        logger.info('Saving rollouts')
        rollouts += new_rollouts
        self._save_rollouts(self._eval_itr, rollouts)

        return rollouts

    def run(self):
        ### Load policy
        policy_fname = self._fm.inference_policy_fname(self._eval_itr)
        if len(glob.glob(os.path.splitext(policy_fname)[0] + '*')) == 0:
            logger.error('Policy for {0} does not exist'.format(policy_fname))
            sys.exit(0)
        logger.info('Restoring policy for itr {0}'.format(self._eval_itr))
        self._policy.restore(policy_fname, train=False)

        ### Load previous eval rollouts
        logger.info('Loading previous eval rollouts')
        rollouts = self._load_rollouts(self._eval_itr)
        logger.info('Loaded {0} rollouts'.format(len(rollouts)))

        self._reset()

        logger.info('')
        logger.info('Rollout {0}'.format(len(rollouts)))
        while len(rollouts) < self._num_rollouts:
            self._step()

            new_rollouts = self._replay_pool_eval.get_recent_rollouts()
            if len(new_rollouts) > 0:
                self._log()
                rollouts = self._save(rollouts, new_rollouts)

                logger.info('')
                logger.info('Rollout {0}'.format(len(rollouts)))
