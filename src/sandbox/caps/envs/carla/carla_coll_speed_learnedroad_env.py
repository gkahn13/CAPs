import numpy as np

from gcg.misc import utils

from sandbox.caps.envs.carla.carla_coll_speed_road_env import CarlaCollSpeedRoadEnv

class CarlaCollSpeedLearnedroadEnv(CarlaCollSpeedRoadEnv):

    #########################
    ### pkls to tfrecords ###
    #########################

    def create_rollout(self, r, labeller, keep_env_infos=False):
        # observations_im
        # observations_vec - fixed
        # actions - fixed
        # rewards - fixed
        # dones - fixed
        # goals
        # steps - fixed
        # env_infos - use and get rid of

        r_new = {
            'observations_vec': r['observations_vec'],
            'actions': r['actions'],
            'dones': r['dones'],
            'steps': r['steps'],
            'env_infos': r['env_infos'] if keep_env_infos else None,
        }

        # observations_im
        obs_im = []
        for t in range(len(r['dones'])-1):
            obs_im_t = []
            for camera_name in self._params['cameras']:
                camera_params = self._params[camera_name]
                if camera_params['include_in_obs']:
                    camera_data = self._convert_camera_data(r['env_infos'][camera_name][t],
                                                            camera_params)
                    obs_im_t.append(camera_data)
            obs_im_t = np.concatenate(obs_im_t, axis=2)
            obs_im_t = utils.imresize(obs_im_t, self.observation_im_space.shape)
            obs_im.append(obs_im_t)
        obs_im.append(obs_im[-1]) # b/c env_infos is short by 1
        obs_im = np.array(obs_im)
        r_new['observations_im'] = obs_im

        r_new['goals'] = np.hstack([r['goals'],
                                    np.nan * np.ones([len(r['goals']), 1], dtype=np.float32),
                                    np.nan * np.ones([len(r['goals']), 1], dtype=np.float32)])


        # goals
        batch_size = 350
        rgb = np.append(r['env_infos']['rgb'], r['env_infos']['rgb'][-1][None], axis=0)
        goals = r_new['goals']
        i = 0
        while i < len(rgb):
            goals[i:i+batch_size] = labeller.label((rgb[i:i+batch_size], None), goals[i:i+batch_size])
            i += batch_size
        r_new['goals'] = goals

        # rewards
        rewards = []
        coll_idx = list(self.observation_vec_spec.keys()).index('coll')
        speed_idx = list(self.observation_vec_spec.keys()).index('speed')
        goal_speed_idx = list(self.goal_spec.keys()).index('speed')
        rightlane_seen_idx = list(self.goal_spec.keys()).index('rightlane_seen')
        rightlane_diff_idx = list(self.goal_spec.keys()).index('rightlane_diff')
        for t in range(len(r_new['dones'])):
            if r_new['observations_vec'][t, coll_idx] > 0:
                r_t = -self.horizon
            else:
                r_t = -abs((r_new['goals'][t, goal_speed_idx] - r_new['observations_vec'][t, speed_idx]) /
                           r_new['goals'][t, goal_speed_idx])

                r_t += 10 * r_new['goals'][t, rightlane_seen_idx] * (1 - abs(r_new['goals'][t, rightlane_diff_idx]))

            rewards.append(r_t)
        rewards = rewards[1:] + [0]
        r_new['rewards'] = np.asarray(rewards, dtype=np.float32)

        return r_new