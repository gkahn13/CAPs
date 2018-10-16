exp_name = 'caps/carla/coll_speed_learnedroad_heading/label_event_cues'
seed = 0
log_level = 'debug'


###################
### Environment ###
###################

from sandbox.caps.envs.carla.carla_coll_speed_learnedroad_heading_env import CarlaCollSpeedLearnedroadHeadingEnv

env_params = {
    'class': CarlaCollSpeedLearnedroadHeadingEnv,
    'kwargs': {
        'run_headless': False,
        'gpu': 0,
        'port': 2070,
        'weather': 2, # cloudy noon
        'player_start_indices': [1], # forces you to turn
        'camera_size': (50, 100),
        'run_server': False
    }
}


################
### Labeller ###
################

from sandbox.caps.labellers.carla.road_labeller import RoadLabeller

labeller_params = {
    'class': RoadLabeller,
    'kwargs': {
        'save_folder': 'caps/carla/learned_road_labeller',
        'pkl_folders': ['caps/carla/coll_speed/dql'],

        'inference_only': True,

        'image_shape': (50, 100, 3),

        'train_steps': int(1e5),
        'eval_every_n_steps': 5,
        'batch_size': 32,
        'log_every_n_steps': int(1e3),
        'save_every_n_steps': int(1e4),
        'lr': 1.e-4,
        'weight_decay': 1e-3,

        'gpu_device': 0,
        'gpu_frac': 0.5,
    }
}


params = {
    'exp_name': exp_name,
    'seed': seed,
    'log_level': log_level,

    'env': env_params,

    'labeller': labeller_params,
}
