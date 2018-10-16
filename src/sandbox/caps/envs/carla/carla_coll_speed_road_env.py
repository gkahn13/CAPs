import numpy as np

from gcg.envs.spaces.box import Box
from gcg.misc import utils

from sandbox.caps.envs.carla.carla_coll_speed_env import CarlaCollSpeedEnv

class CarlaCollSpeedRoadEnv(CarlaCollSpeedEnv):

    def _setup_goal_spec(self):
        goal_spec = super(CarlaCollSpeedRoadEnv, self)._setup_goal_spec()
        goal_spec['rightlane_seen'] = Box(low=0.0, high=1.0) # can you see the right lane?
        goal_spec['rightlane_diff'] = Box(low=-1.0, high=1.0) # distance of right lane from center of image
        return goal_spec

    def _get_goal(self, measurements, sensor_data):
        goal = super(CarlaCollSpeedRoadEnv, self)._get_goal(measurements, sensor_data)
        goal = np.concatenate((goal, np.array([np.nan, np.nan])))
        return goal

    @staticmethod
    def get_rightlane(semantic):
        if len(semantic.shape) == 3:
            semantic = semantic[:, :, 0]

        ROAD_LINES = 6
        ROAD = 7

        combined_mask = np.zeros(semantic.shape, dtype=np.bool)
        rightlane_seen = False
        rightlane_diff = 0

        frac_road = (semantic == ROAD).sum() / np.prod(semantic.shape)
        frac_road_lines = (semantic == ROAD_LINES).sum() / np.prod(semantic.shape)
        if frac_road > 0.05 and frac_road_lines > 0.003:
            road_lines_mask = (np.cumsum(semantic == ROAD_LINES, axis=1) > 0)
            road_mask = (semantic == ROAD)
            combined_mask = np.logical_and(road_lines_mask, road_mask)
            if combined_mask.sum() > 0:
                rightlane_seen = True
                rightlane_diff = (combined_mask.nonzero()[1].mean() - 0.5 * semantic.shape[1]) / (0.5 * semantic.shape[1])

        assert rightlane_diff >= -1. and rightlane_diff <= 1., 'rightlane_diff out of bounds {0}'.format(rightlane_diff)

        return combined_mask, rightlane_seen, rightlane_diff

    def _get_reward(self, measurements, sensor_data):
        if self._is_collision(measurements, sensor_data):
            # collision
            reward = -self.horizon
        else:
            # speed
            pm = measurements.player_measurements
            reward = -abs((self._goal_speed - pm.forward_speed) / self._goal_speed)

            # road
            _, rightlane_seen, rightlane_diff = CarlaCollSpeedRoadEnv.get_rightlane(sensor_data['semantic'].data)
            reward += 10 * rightlane_seen * (1 - abs(rightlane_diff))

        return reward

    #########################
    ### pkls to tfrecords ###
    #########################

    def create_rollout(self, r, labeller):
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
            'env_infos': None,
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
        r_new['observations_im'] = np.array(obs_im)


        # goals
        _, rightlane_seen, rightlane_diff = zip(*[CarlaCollSpeedRoadEnv.get_rightlane(s) for s in r['env_infos']['semantic']])
        # b/c env_infos are one short
        rightlane_seen = list(rightlane_seen) + [rightlane_seen[-1]]
        rightlane_diff = list(rightlane_diff) + [rightlane_diff[-1]]

        r_new['goals'] = np.hstack([r['goals'],
                                    np.array([rightlane_seen], dtype=np.float32).T,
                                    np.array([rightlane_diff], dtype=np.float32).T])

        # rewards
        rewards = []
        coll_idx = list(self.observation_vec_spec.keys()).index('coll')
        speed_idx = list(self.observation_vec_spec.keys()).index('speed')
        goal_speed_idx = list(self.goal_spec.keys()).index('speed')
        for t in range(len(r['dones'])):
            if r['observations_vec'][t, coll_idx] > 0:
                r_t = -self.horizon
            else:
                r_t = -abs((r['goals'][t, goal_speed_idx] - r['observations_vec'][t, speed_idx]) /
                           r['goals'][t, goal_speed_idx])

                r_t += 10 * rightlane_seen[t] * (1 - abs(rightlane_diff[t]))

            rewards.append(r_t)
        rewards = rewards[1:] + [0]
        r_new['rewards'] = np.asarray(rewards, dtype=np.float32)

        return r_new