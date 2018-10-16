import itertools

import numpy as np

from gcg.envs.spaces.box import Box

from sandbox.caps.envs.carla.carla_coll_speed_learnedroad_env import CarlaCollSpeedLearnedroadEnv


class CarlaCollSpeedLearnedroadHeadingEnv(CarlaCollSpeedLearnedroadEnv):

    def __init__(self, params={}):
        super(CarlaCollSpeedLearnedroadHeadingEnv, self).__init__(params=params)

        self._goal_positions = itertools.cycle([
            # np.array([340., 300.]), # right
            # np.array([340., 20.]), # left
            np.array([395., 170.]), # other side
            np.array([5., 325.]), # bottom corner
            np.array([5., 5.]), # top corner
            # np.array([90., 100.]),  # middle
        ])
        self._goal_position = next(self._goal_positions)

    def _setup_goal_spec(self):
        goal_spec = super(CarlaCollSpeedLearnedroadHeadingEnv, self)._setup_goal_spec()
        goal_spec['heading_x'] = Box(low=-1, high=1)
        goal_spec['heading_y'] = Box(low=-1, high=1)
        return goal_spec

    def _get_goal(self, measurements, sensor_data):
        goal = super(CarlaCollSpeedLearnedroadHeadingEnv, self)._get_goal(measurements, sensor_data)

        pm = measurements.player_measurements
        heading_x = self._goal_position[0] - pm.transform.location.x
        heading_y = self._goal_position[1] - pm.transform.location.y
        heading = np.array([heading_x, heading_y])
        heading_norm = np.linalg.norm(heading)
        if heading_norm > 1e-4:
            heading /= heading_norm
        else:
            heading = np.array([0., 0.])
        goal = np.concatenate((goal, heading))

        return goal

    def _get_heading_deviation_reward(self, position, heading, goal_pos):
        u = np.array(goal_pos) - np.array(position)

        if np.linalg.norm(u) < 1e-2:
            return 0

        heading = (np.pi / 180.) * heading
        v = np.array([np.cos(heading), np.sin(heading)])

        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        heading_deviation = np.arccos(np.dot(u, v))
        heading_deviation /= np.pi # normalize to [0, 1]
        # print('heading_deviation: {0:.3f}'.format(heading_deviation))

        return -5. * heading_deviation

    def _get_reward(self, measurements, sensor_data):
        reward = super(CarlaCollSpeedLearnedroadHeadingEnv, self)._get_reward(measurements, sensor_data)

        if not self._is_collision(measurements, sensor_data):
            pm = measurements.player_measurements
            position = np.array([pm.transform.location.x, pm.transform.location.y])
            heading = pm.transform.rotation.yaw
            reward += self._get_heading_deviation_reward(position, heading, self._goal_position)

        return reward

    def _get_done(self, measurements, sensor_data):
        done = super(CarlaCollSpeedLearnedroadHeadingEnv, self)._get_done(measurements, sensor_data)

        pm = measurements.player_measurements
        distance_to_goal = np.linalg.norm(np.array([pm.transform.location.x, pm.transform.location.y]) -
                                          self._goal_position)
        done = done or (distance_to_goal < 10)

        return done

    ####################
    ### Step / Reset ###
    ####################

    def reset(self, player_start_idx=None):
        obs, goal = super(CarlaCollSpeedLearnedroadHeadingEnv, self).reset(player_start_idx=player_start_idx)

        self._goal_position = next(self._goal_positions)

        return obs, goal

    ###############
    ### Logging ###
    ###############

    def _update_log_stats(self):
        super(CarlaCollSpeedLearnedroadHeadingEnv, self)._update_log_stats()

        pm = self._last_measurements.player_measurements

        # distance to goal
        distance_to_goal = np.linalg.norm(np.array([pm.transform.location.x, pm.transform.location.y]) -
                                          self._goal_position)
        self._log_stats['DistanceToGoal'].append(distance_to_goal)
        if self._get_done(self._last_measurements, self._last_sensor_data):
            self._log_stats['IsCollision'] = self._is_collision(self._last_measurements, self._last_sensor_data)
            self._log_stats['FinalDistanceToGoal'].append(distance_to_goal)

    #########################
    ### pkls to tfrecords ###
    #########################

    def create_rollout(self, r, labeller):
        r_new = super(CarlaCollSpeedLearnedroadHeadingEnv, self).create_rollout(r, labeller, keep_env_infos=True)

        # goals
        heading_x = []
        heading_y = []
        for t in range(len(r_new['dones']) - 1):
            x_pos = r_new['env_infos']['pos_x'][t]
            y_pos = r_new['env_infos']['pos_y'][t]

            x_pos_ahead = r_new['env_infos']['pos_x'][-1]
            y_pos_ahead = r_new['env_infos']['pos_y'][-1]

            heading_t = np.array([x_pos_ahead - x_pos, y_pos_ahead - y_pos])
            heading_t_norm = np.linalg.norm(heading_t)
            if heading_t_norm > 1e-4:
                heading_t /= heading_t_norm
            else:
                heading_t = np.array([0., 0.])

            heading_x.append(heading_t[0])
            heading_y.append(heading_t[1])

        heading_x.append(heading_x[-1])
        heading_y.append(heading_y[-1])

        r_new['goals'] = np.hstack([r_new['goals'],
                                    np.array([heading_x]).T,
                                    np.array([heading_y]).T])

        assert r_new['goals'][:, -2].min() >= -1.0
        assert r_new['goals'][:, -2].max() <= 1.0
        assert r_new['goals'][:, -1].min() >= -1.0
        assert r_new['goals'][:, -1].max() <= 1.0

        # rewards
        coll_idx = list(self.observation_vec_spec.keys()).index('coll')
        heading_idx = list(self.observation_vec_spec.keys()).index('heading')
        rewards = []
        for t in range(len(r_new['dones']) - 1):
            r_t = r_new['rewards'][t]

            if r_new['observations_vec'][t, coll_idx] < 1e-4:
                x_pos = r_new['env_infos']['pos_x'][t]
                y_pos = r_new['env_infos']['pos_y'][t]
                x_goal_pos_delta = r_new['goals'][t, -2]
                y_goal_pos_delta = r_new['goals'][t, -1]

                position = np.array([x_pos, y_pos])
                heading = r_new['observations_vec'][t, heading_idx]
                goal_pos = np.array([x_pos + x_goal_pos_delta, y_pos + y_goal_pos_delta])
                r_t += self._get_heading_deviation_reward(position, heading, goal_pos)

            rewards.append(r_t)
        rewards.append(r_new['rewards'][-1])
        r_new['rewards'] = np.array(rewards)

        return r_new