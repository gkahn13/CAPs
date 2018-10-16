class DummyPool(object):
    """
    Does nothing, useful if reading data some other way (e.g., tfrecords)
    """

    def __init__(self, **kwargs):
        self._current_rollout_len = 0

    def __len__(self):
        return 0

    @property
    def size(self):
        return 0

    ###################
    ### Add to pool ###
    ###################

    def store_observation(self, step, observation, goal, use_labeller=True):
        self._current_rollout_len += 1

    def encode_recent_observation(self):
        pass

    def store_effect(self, action, reward, done, env_info, flatten_action=True, update_log_stats=True):
        if done:
            self._current_rollout_len = 0

    def force_done(self):
        pass

    def store_rollouts(self, rlist, max_to_add=None):
        pass

    def update_priorities(self, indices, priorities):
        pass

    ########################
    ### Remove from pool ###
    ########################

    def trash_current_rollout(self):
        l = self._current_rollout_len
        self._current_rollout_len = 0
        return l

    ########################
    ### Sample from pool ###
    ########################

    def can_sample(self, batch_size=1):
        return False

    def sample(self, batch_size, include_env_infos=False):
        raise NotImplementedError

    def sample_all_generator(self, batch_size, include_env_infos=False):
        pass

    def sample_rollouts(self, num_rollouts):
        raise NotImplementedError

    ###############
    ### Logging ###
    ###############

    def log(self, prefix=''):
        pass

    def get_recent_rollouts(self):
        return []

    @property
    def finished_storing_rollout(self):
        return True
