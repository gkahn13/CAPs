import os
from collections import OrderedDict, defaultdict
import subprocess
import random
import signal
from fractions import Fraction

import numpy as np

from gcg.envs.env import Env
from gcg.envs.env_spec import EnvSpec
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger

from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.sensor import Camera


class CarlaCollSpeedEnv(Env):
    RETRIES = 4
    WIDTH_TO_HEIGHT_RATIO = 2
    CAMERA_HEIGHT = 300

    def __init__(self, params={}):
        params.setdefault('carla_path', '$CARLAPATH')
        params.setdefault('gpu', 0)
        params.setdefault('run_headless', False)
        params.setdefault('run_server', True)

        params.setdefault('map', '/Game/Maps/Town01')
        params.setdefault('fps', 4)
        params.setdefault('host', 'localhost')
        params.setdefault('port', 2000)

        params.setdefault('cameras', ['rgb', 'depth', 'semantic'])
        params.setdefault('camera_size', (50, 100))
        params.setdefault('rgb',
                          {
                              # for carla
                              'postprocessing': 'SceneFinal',
                              'position': (1.6, 0, 1.40),
                              'fov': 120,  # x position and fov are correlated

                              # for us
                              'include_in_obs': True,
                              'grayscale': True,

                          })
        params.setdefault('depth',
                          {
                              # for carla
                              'postprocessing': 'Depth',
                              'position': (1.6, 0, 1.40),
                              'fov': 120,  # x position and fov are correlated

                              # for us
                              'include_in_obs': False,
                          })
        params.setdefault('semantic',
                          {
                              # for carla
                              'postprocessing': 'SemanticSegmentation',
                              'position': (1.6, 0, 1.40),
                              'fov': 120,  # x position and fov are correlated

                              # for us
                              'include_in_obs': False,
                          })
        params.setdefault('number_of_vehicles', 0)
        params.setdefault('number_of_pedestrians', 0)
        params.setdefault('weather', -1)

        params.setdefault('player_start_indices', None)
        params.setdefault('horizon', 1000)
        params.setdefault('goal_speed', 7.0) # 7 m/s == 25 km/hr == 15 mph
        params.setdefault('steps_after_reset', 15)

        self._player_start_indices = params['player_start_indices']
        self._horizon = params['horizon']
        self._goal_speed = params['goal_speed']
        self._steps_after_reset = params['steps_after_reset']

        self._params = params
        self._carla_server_process = None
        self._carla_client = None
        self._carla_settings = None
        self._carla_scene = None
        self.action_spec = None
        self.action_selection_spec = None
        self.observation_vec_spec = None
        self.goal_spec = None
        self.action_space = None
        self.action_selection_space = None
        self.observation_im_space = None
        self.spec = None
        self._last_measurements = None
        self._last_sensor_data = None
        self._t = 0
        self._curr_episode_max_speed = 0
        self._curr_episode_coll_other = 0
        self._log_stats = defaultdict(list)

        assert(Fraction(*self._params['camera_size']).numerator == 1 and \
               Fraction(*self._params['camera_size']).denominator == CarlaCollSpeedEnv.WIDTH_TO_HEIGHT_RATIO)

        ### setup
        self._setup_spec()
        if params['run_server']:
            self._setup_carla()

    #############
    ### Setup ###
    #############

    def _clear_carla_server(self):
        try:
            if self._carla_client is not None:
                self._carla_client.disconnect()
        except Exception as e:
            logger.warn('Error disconnecting client: {}'.format(e))
        self._carla_client = None

        if self._carla_server_process:
            pgid = os.getpgid(self._carla_server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self._carla_server_process = None

    def _setup_carla_server(self):
        assert (self._carla_server_process is None)

        server_bash_path = os.path.abspath(os.path.join(os.path.expandvars(self._params['carla_path']),
                                                        'CarlaUE4.sh'))
        map = self._params['map']
        fps = self._params['fps']
        port = self._params['port']

        carla_settings_path = '/tmp/CarlaSettings.ini'
        with open(carla_settings_path, 'w') as f:
            f.writelines(['[CARLA/Server]\n', 'UseNetworking=true\n', 'ServerTimeOut=1000000\n'])
        carla_settings_path = os.path.relpath(carla_settings_path, os.path.abspath(__file__))

        assert((map == '/Game/Maps/Town01') or (map == '/Game/Maps/Town02'))

        # kill_cmd = 'pkill -9 -f {0}'.format(server_bash_path)
        # subprocess.Popen(kill_cmd, shell=True)

        server_env = os.environ.copy()
        cmd = []
        if self._params['run_headless']:
            server_env['DISPLAY'] = ':1'
            cmd += ['vglrun',
                    '-d',
                    ':0.{0}'.format(self._params['gpu'])]

        cmd += [server_bash_path,
                map,
                '-carla-server',
                '-benchmark',
                '-fps={0}'.format(fps),
                '-carla-settings="{0}"'.format(carla_settings_path),
                '-carla-world-port={0}'.format(port),
                '-carla-no-hud',
                '-windowed',
                '-ResX=400',
                '-ResY=300'
                ]
        print(' '.join(cmd))
        self._carla_server_process = subprocess.Popen(cmd,
                                                      stdout=open(os.devnull, 'w'),
                                                      preexec_fn=os.setsid,
                                                      env=server_env)

    def _setup_carla_client(self):
        carla_client = CarlaClient(self._params['host'], self._params['port'], timeout=None)
        carla_client.connect()

        ### create initial settings
        carla_settings = CarlaSettings()
        carla_settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=self._params['number_of_vehicles'],
            NumberOfPedestrians=self._params['number_of_pedestrians'],
            WeatherId=self._params['weather'])
        carla_settings.randomize_seeds()

        ### add cameras
        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            camera_postprocessing = camera_params['postprocessing']
            camera = Camera(camera_name, PostProcessing=camera_postprocessing)
            camera.set_image_size(CarlaCollSpeedEnv.CAMERA_HEIGHT * CarlaCollSpeedEnv.WIDTH_TO_HEIGHT_RATIO, CarlaCollSpeedEnv.CAMERA_HEIGHT)
            camera.set_position(*camera_params['position'])
            camera.set(**{'FOV': camera_params['fov']})

            carla_settings.add_sensor(camera)

        ### load settings
        carla_scene = carla_client.load_settings(carla_settings)

        self._carla_client = carla_client
        self._carla_settings = carla_settings
        self._carla_scene = carla_scene

    def _setup_carla(self):
        self._clear_carla_server()
        self._setup_carla_server()
        self._setup_carla_client()

    def _setup_spec(self):
        action_spec = OrderedDict()
        action_selection_spec = OrderedDict()
        observation_vec_spec = OrderedDict()

        action_spec['steer'] = Box(low=-1., high=1.)
        action_spec['motor'] = Box(low=-1., high=1.)
        action_space = Box(low=np.array([v.low[0] for k, v in action_spec.items()]),
                                high=np.array([v.high[0] for k, v in action_spec.items()]))

        action_selection_spec['steer'] = Box(low=-1., high=1.)
        action_selection_spec['motor'] = Box(low=0.3, high=1.) # min 0.3--> 1.4m/s == 4km/h, need 1km/h to detect collisions
        action_selection_space = Box(low=np.array([v.low[0] for k, v in action_selection_spec.items()]),
                                          high=np.array([v.high[0] for k, v in action_selection_spec.items()]))

        assert (np.logical_and(action_selection_space.low >= action_space.low,
                               action_selection_space.high <= action_space.high).all())

        num_channels = 0
        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            assert(camera_params['postprocessing'] is not None)
            if camera_params['include_in_obs']:
                if camera_params['postprocessing'] == 'SceneFinal':
                    num_channels += 1 if camera_params.get('grayscale', False) else 3
                else:
                    num_channels += 1
        observation_im_space = Box(low=0, high=255, shape=list(self._params['camera_size']) + [num_channels])

        observation_vec_spec['coll'] = Discrete(1)
        observation_vec_spec['coll_car'] = Discrete(1)
        observation_vec_spec['coll_ped'] = Discrete(1)
        observation_vec_spec['coll_oth'] = Discrete(1)
        observation_vec_spec['heading'] = Box(low=-180., high=180.)
        observation_vec_spec['speed'] = Box(low=-30., high=30.)
        observation_vec_spec['accel_x'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_y'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_z'] = Box(low=-100., high=100.)

        goal_spec = self._setup_goal_spec()

        self.action_spec, self.action_selection_spec, self.observation_vec_spec, self.goal_spec, \
        self.action_space, self.action_selection_space, self.observation_im_space = \
            action_spec, action_selection_spec, observation_vec_spec, goal_spec, \
                action_space, action_selection_space, observation_im_space

        self.spec = EnvSpec(
            observation_im_space=observation_im_space,
            action_space=action_space,
            action_selection_space=action_selection_space,
            observation_vec_spec=observation_vec_spec,
            action_spec=action_spec,
            action_selection_spec=action_selection_spec,
            goal_spec=goal_spec)

    def _setup_goal_spec(self):
        goal_spec = OrderedDict()
        goal_spec['speed'] = Box(low=-30.0, high=30.0)
        return goal_spec

    ###############
    ### Get/Set ###
    ###############

    def _execute_action(self, a):
        steer = a[0]
        throttle = abs(a[1])
        reverse = (a[1] < 0)
        brake = 0

        self._carla_client.send_control(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=reverse)

    def _execute_null_action(self):
        self._carla_client.send_control(
            steer=0,
            throttle=0,
            brake=0,
            hand_brake=False,
            reverse=False)

    def _is_collision(self, measurements, sensor_data):
        pm = measurements.player_measurements

        collision_other_kg = (pm.collision_other - self._curr_episode_coll_other) / max(abs(pm.forward_speed), 0.01)

        coll = ((pm.collision_vehicles > 0) or (pm.collision_pedestrians > 0) or (collision_other_kg > 20))
        if self._last_measurements:
            coll = coll or (abs(pm.forward_speed) < 0.8 and self._curr_episode_max_speed >= 1.0)
        return coll

    def _convert_camera_data(self, camera_data, camera_params):
        if len(camera_data.shape) == 2:
            camera_data = np.expand_dims(camera_data, axis=2)

        # rgb
        if camera_params.get('grayscale', False):
            def rgb2gray(rgb):
                return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

            camera_data = np.expand_dims(rgb2gray(camera_data), axis=2)

        if camera_params['postprocessing'] == 'Depth':
            min_range, max_range = 1e-4, 1
            camera_data = np.clip(camera_data, min_range, max_range)
            camera_data = np.log(camera_data)
            camera_data = 255 * (camera_data - np.log(min_range)) / (np.log(max_range) - np.log(min_range))
            assert (camera_data.min() >=0 and camera_data.max() <= 255)

        camera_data = camera_data.astype(np.uint8)

        return camera_data

    def _get_observation_im(self, measurements, sensor_data):
        obs_im = []

        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            if camera_params['include_in_obs']:
                camera_data = self._convert_camera_data(sensor_data[camera_name].data,
                                                        camera_params)
                obs_im.append(camera_data)

        return np.concatenate(obs_im, axis=2)

    def _get_observation_vec(self, measurements, sensor_data):
        pm = measurements.player_measurements

        obs_vec_d = {}
        obs_vec_d['heading'] = pm.transform.rotation.yaw
        obs_vec_d['speed'] = pm.forward_speed
        obs_vec_d['accel_x'] = pm.acceleration.x
        obs_vec_d['accel_y'] = pm.acceleration.y
        obs_vec_d['accel_z'] = pm.acceleration.z
        obs_vec_d['coll_car'] = (pm.collision_vehicles > 0)
        obs_vec_d['coll_ped'] = (pm.collision_pedestrians > 0)
        obs_vec_d['coll_oth'] = (pm.collision_other > 0)
        obs_vec_d['coll'] = self._is_collision(measurements, sensor_data)

        # print('t: {0}, coll: {1}, speed: {2:.2f}, accel_x: {3:.2f}'.format(self._t,
        #                                                                           obs_vec_d['coll'],
        #                                                                           obs_vec_d['speed'],
        #                                                                           obs_vec_d['accel_x']))

        obs_vec = np.array([obs_vec_d[k] for k in self.observation_vec_spec.keys()])
        return obs_vec

    def _get_goal(self, measurements, sensor_data):
        return np.array([self._goal_speed], dtype=np.float32)

    def _get_reward(self, measurements, sensor_data):
        if self._is_collision(measurements, sensor_data):
            reward = -self.horizon
        else:
            pm = measurements.player_measurements
            reward = -abs((self._goal_speed - pm.forward_speed) / self._goal_speed)

        return reward

    def _get_done(self, measurements, sensor_data):
        return self._is_collision(measurements, sensor_data)

    def _get_env_info(self, measurements, sensor_data):
        env_info = {}

        # measurements
        pm = measurements.player_measurements
        env_info['pos_x'] = pm.transform.location.x
        env_info['pos_y'] = pm.transform.location.y
        env_info['pos_z'] = pm.transform.location.z
        env_info['intersection_otherlane'] = pm.intersection_otherlane
        env_info['intersection_offroad'] = pm.intersection_offroad

        # sensors
        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            env_info[camera_name] = self._convert_camera_data(sensor_data[camera_name].data,
                                                              {**camera_params, 'grayscale': False})
            assert(tuple(env_info[camera_name].shape[:2]) == \
                   (CarlaCollSpeedEnv.CAMERA_HEIGHT, CarlaCollSpeedEnv.CAMERA_HEIGHT * CarlaCollSpeedEnv.WIDTH_TO_HEIGHT_RATIO))

        return env_info

    ####################
    ### Step / Reset ###
    ####################

    def _get(self, measurements=None, sensor_data=None):
        if measurements is None or sensor_data is None:
            measurements, sensor_data = self._carla_client.read_data()

        self._curr_episode_max_speed = max(self._curr_episode_max_speed,
                                           abs(measurements.player_measurements.forward_speed))


        next_observation = (self._get_observation_im(measurements, sensor_data),
                            self._get_observation_vec(measurements, sensor_data))
        goal = self._get_goal(measurements, sensor_data)
        reward = self._get_reward(measurements, sensor_data)
        done = self._get_done(measurements, sensor_data)
        env_info = self._get_env_info(measurements, sensor_data)

        self._curr_episode_coll_other = measurements.player_measurements.collision_other

        self._last_measurements = measurements
        self._last_sensor_data = sensor_data

        return next_observation, goal, reward, done, env_info

    def step(self, action):
        try:
            self._execute_action(action)
            next_observation, goal, reward, done, env_info = self._get()
        except Exception as e:
            logger.warn('CarlaCollSpeedEnv: Error during step: {}'.format(e))
            self._setup_carla()
            next_observation, goal, reward, done, env_info = self._get(measurements=self._last_measurements,
                                                                       sensor_data=self._last_sensor_data)
            done = True

        self._t += 1
        self._update_log_stats()

        return next_observation, goal, reward, done, env_info

    def reset(self, player_start_idx=None):
        number_of_player_starts = len(self._carla_scene.player_start_spots)
        if player_start_idx is None:
            if self._player_start_indices is None:
                player_start_idx = np.random.randint(number_of_player_starts)
            else:
                player_start_idx = random.choice(self._player_start_indices)
        else:
            player_start_idx = int(player_start_idx) % number_of_player_starts
        assert ((0 <= player_start_idx) and (player_start_idx < number_of_player_starts))

        error = None
        for _ in range (CarlaCollSpeedEnv.RETRIES):
            try:
                self._t = 0
                self._curr_episode_max_speed = 0
                self._curr_episode_coll_other = 0
                self._carla_client.start_episode(player_start_idx)
                for step in range(self._steps_after_reset):
                    self._execute_null_action()
                    self._carla_client.read_data()
                next_observation, goal, reward, done, env_info = self._get()
                return next_observation, goal
            except Exception as e:
                logger.warn('CarlaCollSpeedEnv: start episode error: {}'.format(e))
                self._setup_carla()
                error = e
        else:
            logger.critical('CarlaCollSpeedEnv: Failed to restart after {0} attempts'.format(CarlaCollSpeedEnv.RETRIES))
            raise error

    @property
    def horizon(self):
        return self._horizon

    ###############
    ### Logging ###
    ###############

    def _update_log_stats(self):
        pm = self._last_measurements.player_measurements

        self._log_stats['Speed'].append(pm.forward_speed)
        self._log_stats['Intersection_otherlane'].append(pm.intersection_otherlane)
        self._log_stats['Intersection_offroad'].append(pm.intersection_offroad)

    def log(self, prefix=''):
        for key in sorted(self._log_stats.keys()):
            logger.record_tabular('{0}{1}Mean'.format(prefix, key), np.mean(self._log_stats[key]))
            logger.record_tabular('{0}{1}Std'.format(prefix, key), np.std(self._log_stats[key]))
        self._log_stats = defaultdict(list)

    #########################
    ### pkls to tfrecords ###
    #########################

    def create_rollout(self, r, labeller):
        r_new = {
            'observations_im': r['observations_im'],
            'observations_vec': r['observations_vec'],
            'actions': r['actions'],
            'dones': r['dones'],
            'steps': r['steps'],
            'goals': r['goals'],
            'env_infos': None,
        }

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

            rewards.append(r_t)
        rewards = rewards[1:] + [0]
        r_new['rewards'] = np.asarray(rewards, dtype=np.float32)

        return r_new

def test_reset(env):
    N = 6
    import time
    start = time.time()
    for _ in range(N):
        env.reset()
    elapsed = time.time() - start
    print('avg time per reset: {0}'.format(elapsed / float(N)))

def test_collision(env):
    import time
    env.reset()
    start = time.time()
    t = 0
    while True:
        t += 1
        (obs_im, obs_vec), goal, reward, done, env_info = env.step([0.0, 0.4])

        # for k in ('intersection_offroad', 'intersection_otherlane'):
        #     print('{0}: {1}'.format(k, env_info[k]))
        # input('')

        # import IPython; IPython.embed()

        # import matplotlib.pyplot as plt
        # plt.imshow(obs_im)
        # plt.show()

        if t % 10 == 0:
            print('time per step: {0}'.format((time.time() - start) / 10.))
            start = time.time()

        if done:
            t = 0
            break
            # env.reset()

if __name__ == '__main__':
    logger.setup(display_name='CarlaCollSpeedEnv', log_path='/tmp/log.txt', lvl='debug')
    env = CarlaCollSpeedEnv(params={
        'port': 2040,
        'player_start_indices': [30],
        'number_of_pedestrians': 150,
        'number_of_vehicles': 0,
        'weather': 2,
    })
    test_collision(env)
    # test_reset(env)
    import IPython; IPython.embed()