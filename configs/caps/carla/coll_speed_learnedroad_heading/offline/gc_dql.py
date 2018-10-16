exp_name = 'caps/carla/coll_speed_learnedroad_heading/offline/gc_dql'
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

env_eval_params = { # if None, env_eval = env
    'class': env_params['class'],
    'kwargs': {
        **env_params['kwargs'],
        'run_server': True
    }
}

###################
### Replay pool ###
###################

from gcg.replay_pools.dummy_pool import DummyPool

rp_params = {
    'class': DummyPool,
    'kwargs': {
    }
}

rp_eval_params = {
    'class': DummyPool,
    'kwargs': {
    }
}


################
### Labeller ###
################

labeller_params = None


#################
### Algorithm ###
#################

from gcg.algos.gcg_train_tfrecord import GCGtrainTfrecord

from gcg.exploration_strategies.gaussian_strategy import GaussianStrategy
from gcg.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

alg_params = {
    'class': GCGtrainTfrecord,
    'kwargs': {
        ### Offpolicy data ###

        'offpolicy': ['/home/caps-user/caps/data/caps/carla/coll_speed_learnedroad_heading/label_event_cues'],  # folder path containing .pkl/.tfrecord files with rollouts
        'num_offpolicy': None,  # number of offpolicy datapoints to load
        'init_train_ckpt': None, # initial training checkpoint model to load from
        'init_inference_ckpt': None, # initial inference checkpoint model to load from


        ### Steps ###

        'total_steps': 5.e+6,  # corresponding to number of env.step(...) calls

        'sample_after_n_steps': -1,
        'onpolicy_after_n_steps': 4.e+3,  # take random actions until this many steps is reached

        'learn_after_n_steps': -1,  # when to start training the model
        'train_every_n_steps': 1, # number of calls to model.train per env.step (if fractional, multiple trains per step)
        'eval_every_n_steps': 5,  # how often to evaluate policy in env_eval
        'rollouts_per_eval': 1,  # how many rollouts to evaluate per eval_every_n_steps

        'update_target_after_n_steps': -1,  # after which the target network can be updated
        'update_target_every_n_steps': 5.e+3,  # how often to update target network

        'save_every_n_steps': 1.e+4, # how often to save experiment data
        'save_async': False, # save files in the background?
        'log_every_n_steps': 1.e+3,  # how often to print log information

        'batch_size': 32,  # per training step

        ### Exploration ###

        'exploration_strategies': [
            {
                'class': GaussianStrategy,
                'params': {
                    # endpoints: [[step, value], [step, value], ...]
                    'endpoints': [[0, 0.25], [8.e+4, 0.05], [24.e+4, 0.005]],
                    'outside_value': 0.005
                },
            },
            {
                'class': EpsilonGreedyStrategy,
                'params': {
                    # endpoints: [[step, value], [step, value], ...]
                    'endpoints': [[0, 1.0], [1.e+3, 1.0], [8.e+4, 0.1], [16.e+4, 0.01]],
                    'outside_value': 0.01
                },
            },
        ],
    }
}


##############
### Policy ###
##############

from gcg.policies.gcg_policy_tfrecord import GCGPolicyTfrecord
import tensorflow as tf
from gcg.tf.layers.cnn.convolution import Convolution
from gcg.tf.layers.fullyconnectednn.fully_connected import FullyConnected
from gcg.tf.layers.rnn.rnn_cell import DpMulintLSTMCell

def bhat_label_func(rewards, dones, goals, target_obs_vec, gamma,
                    future_goals, target_yhats, target_bhats, target_values):
    import numpy as np
    H = rewards.get_shape()[1].value
    gammas = np.power(gamma, np.arange(1, H+1))
    bhat_label = rewards + (1 - dones[:, 1:]) * gammas * target_values['value']
    return bhat_label

policy_params = {
    'class': GCGPolicyTfrecord,  # <GCGPolicy> model class

    'kwargs': {
        'N': 1,  # label horizon
        'H': 1,  # model horizon
        'gamma': 0.95,  # discount factor
        'obs_history_len': 4,  # number of previous observations to concatenate (inclusive)

        'use_target': True,  # target network?

        'goals_to_input': ['speed', 'heading_x', 'heading_y'], # which goals to input?

        # actions, yhats, bhats, values, goals are available
        # action_selection_value
        # :param actions
        # :param yhats
        # :param bhats
        # :param values
        # :param goals
        'action_selection_value': lambda actions, yhats, bhats, values, goals: values['value'],

        'outputs': [
            {
                'name': 'value',
                # yhat
                # :param pre_yhat: [batch_size, H]
                # :param obs_vec: {name : [batch_size, 1]}
                'yhat': None,

                # yhat_label
                # :param rewards: [batch_size, N]
                # :param dones: [batch_size, N+1]
                # :param goals: {name: [batch_size, 1]}
                # :param future_goals: {name: [batch_size, N]}
                # :param target_obs_vec: [batch_size, N]
                # :param gamma: scalar
                'yhat_label': None,

                # yhat training cost
                'yhat_loss': None, # <mse / huber / xentropy>
                'yhat_loss_weight': None, # how much to weight this loss compared to other losses
                'yhat_loss_use_pre': None,  # use the pre-activation for the loss? needed for xentropy
                'yhat_loss_xentropy_posweight': None,  # larger value --> false negatives cost more

                # bhat
                # :param pre_bhat: [batch_size, H]
                # :param obs_vec: {name: [batch_size, 1]}
                'bhat': lambda pre_bhat, obs_vec: pre_bhat,

                # bhat_label
                # :param rewards: [batch_size, N]
                # :param dones: [batch_size, N+1]
                # :param goals: {name: [batch_size, 1]}
                # :param target_obs_vec: {name: [batch_size, H]}
                # :param gamma: scalar
                # :param future_goals: {name: [batch_size, N]}
                # :param target_yhats: {name: [batch_size, N, H]}
                # :param target_bhats: {name: [batch_size, N, H]}
                # :param target_values: {name: [batch_size, N]}
                'bhat_label': bhat_label_func,

                # bhat training cost
                'bhat_loss': 'mse', # <mse / huber / xentropy>
                'bhat_loss_weight': 1.0, # how much to weight this loss compared to other losses
                'bhat_loss_use_pre': False,  # use the pre-activation for the loss? needed for xentropy
                'bhat_loss_xentropy_posweight': None,  # larger value --> false negatives cost more

                # value
                # :param yhats: {name: [batch_size, H]}
                # :param bhats: {name: [batch_size, H]}
                # :param goals: {name: [batch_size, 1]}
                # :param gamma
                'value': lambda yhats, bhats, goals, gamma: tf.reduce_mean(bhats['value'], axis=1),

                # do you train RNN beyond dones?
                'clip_with_done': True,
            }
        ],

        ### Action selection

        'get_action_test': {  # how to select actions at test time (i.e., when gathering samples)
            'H': 1,
            'type': 'cem',  # <random/cem> action selection method
            'random': {
                'K': 4096,
            },
            'cem': {
                'init': {
                    'M_init': 4096,
                    'M': 1024,
                    'itrs': 4,
                },
                'warm_start': {
                    'M_init': 4096,
                    'M': 1024,
                    'itrs': 4,
                },
                'K': 128,
                'eps': 1.e-4,
                'K_last_top': 0
            }
        },

        'get_action_target': {
            'H': 1,
            'type': 'random',  # <random>

            'random': {
                'K': 100,
            },
        },

        ### Network architecture

        'image_graph': {  # CNN
            'conv_class': Convolution,
            'conv_args': {},
            'filters': [64, 32, 32, 32],
            'kernels': [8, 4, 3, 3],
            'strides': [4, 2, 2, 2],
            'padding': 'SAME',
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True
        },

        'observation_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [256],
            'output_dim': 128,  # this is the hidden size of the rnn
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True,
        },

        'action_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [16],
            'output_dim': 16,
            'hidden_activation': tf.nn.relu,
            'output_activation': tf.nn.relu,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True,
        },

        'rnn_graph': {
            'rnncell_class': DpMulintLSTMCell,
            'rnncell_args': {},
            'num_cells': 1,
            'state_tuple_size': 2, # 1 for standard cells, 2 for LSTM
            'trainable': True
        },

        'output_graph': {  # fully connected
            'fullyconnected_class': FullyConnected,
            'fullyconnected_args': {},
            'hidden_layers': [16],
            # 'output_dim': None, # is determined by yhat / bhat
            'hidden_activation': tf.nn.relu,
            'output_activation': None,
            'normalizer_fn': None,
            'normalizer_params': None,
            'trainable': True,
        },

        ### Training

        'optimizer': 'adam',  # <adam/sgd>
        'weight_decay': 0.5,  # L2 regularization
        'lr_schedule': {  # learning rate schedule
            'endpoints': [],
            'outside_value': 1.e-4,
        },
        'grad_clip_norm': 10,  # clip the gradient magnitude

        ### Device
        'gpu_device': 0,
        'gpu_frac': 0.4,
        'seed': 0
    },
}

##################
### Evaluation ###
##################

from gcg.algos.gcg_eval import GCGeval

eval_params = {
    'class': GCGeval,
    'kwargs': {
    }
}


params = {
    'exp_name': exp_name,
    'seed': seed,
    'log_level': log_level,

    'env': env_params,
    'env_eval': env_eval_params,

    'replay_pool': rp_params,
    'replay_pool_eval': rp_eval_params,

    'labeller': labeller_params,

    'alg': alg_params,

    'policy': policy_params,

    'eval': eval_params
}
