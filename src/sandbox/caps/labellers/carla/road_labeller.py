import os, random
import multiprocessing
import PIL

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gcg.labellers.labeller import Labeller
from gcg.data.logger import logger
from gcg.data.file_manager import FileManager, DATA_DIR
from gcg.data import mypickle
from gcg.misc import utils

from sandbox.caps.envs.carla.carla_coll_speed_road_env import CarlaCollSpeedRoadEnv
from sandbox.caps.labellers.traversability.train.traversability_graph import TraversabilityGraph

class RoadLabeller(Labeller):

    def __init__(self, env_spec, policy, **kwargs):
        super(RoadLabeller, self).__init__(env_spec, policy)

        self._pkl_folders = kwargs['pkl_folders']
        self._image_shape = list(kwargs['image_shape'])
        self._train_steps = int(kwargs['train_steps'])
        self._eval_every_n_steps = int(kwargs['eval_every_n_steps'])
        self._log_every_n_steps = int(kwargs['log_every_n_steps'])
        self._save_every_n_steps = int(kwargs['save_every_n_steps'])
        self._inference_only = kwargs.get('inference_only', False)

        self._trav_graph = TraversabilityGraph(image_shape=self._image_shape,
                                               save_folder=os.path.join(DATA_DIR, kwargs['save_folder']),
                                               batch_size=kwargs['batch_size'],
                                               weight_decay=kwargs['weight_decay'],
                                               lr=kwargs['lr'],
                                               gpu_device=kwargs['gpu_device'],
                                               gpu_frac=kwargs['gpu_frac'],
                                               inference_only=kwargs.get('inference_only', False))
        if self._trav_graph.is_ckpt:
            self._trav_graph.restore()
        else:
            assert not self._inference_only

    def label(self, observations, goals):
        """
        :return goals
        """
        observations_im, _ = observations
        pred_labels = self._trav_graph.get_model_outputs(observations_im)

        rightlane_seen = []
        rightlane_diff = []
        for combined_mask_t in (pred_labels > 0.5):
            rightlane_seen_t = ((combined_mask_t.sum() / np.prod(combined_mask_t.shape)) > 0.02)
            if rightlane_seen_t:
                rightlane_diff_t = (combined_mask_t.nonzero()[1].mean() - 0.5 * combined_mask_t.shape[1]) \
                                    / (0.5 * combined_mask_t.shape[1])
                assert abs(rightlane_diff_t) <= 1.0, 'rightlane_diff should be in [-1, 1], but is {0}'.format(rightlane_diff_t)
            else:
                rightlane_diff_t = 0.

            rightlane_seen.append(rightlane_seen_t)
            rightlane_diff.append(rightlane_diff_t)

        new_goals = np.copy(goals)
        goal_keys = list(self._env_spec.goal_spec.keys())
        new_goals[:, goal_keys.index('rightlane_seen')] = rightlane_seen
        new_goals[:, goal_keys.index('rightlane_diff')] = rightlane_diff

        return new_goals

    ################
    ### Training ###
    ################

    def train(self):
        logger.setup(display_name=self._trav_graph.save_folder,
                     log_path=os.path.join(self._trav_graph.save_folder, 'log.txt'),
                     lvl='debug')

        # self._create_training_data()
        self._train_model()
        # self._eval_model()

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _write_tfrecord(pkl_fname, tfrecord_fname, obs_shape):
        writer = tf.python_io.TFRecordWriter(tfrecord_fname)

        rollouts = mypickle.load(pkl_fname)['rollouts']
        for r in rollouts:
            images = r['env_infos']['rgb']
            semantics = r['env_infos']['semantic']

            for image, semantic in zip(images, semantics):
                rightlane_mask = CarlaCollSpeedRoadEnv.get_rightlane(semantic)[0].astype(np.uint8)

                image = utils.imresize(image, obs_shape)
                rightlane_mask = utils.imresize(rightlane_mask[..., None],
                                                obs_shape[:2] + [1],
                                                PIL.Image.BILINEAR)[..., -1]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': RoadLabeller._bytes_feature(image.tostring()),
                            'label': RoadLabeller._bytes_feature(rightlane_mask.tostring()),
                        }))
                writer.write(example.SerializeToString())

        writer.close()

    def _create_training_data(self, num_processes=10):
        assert len([f for f in os.listdir(self._trav_graph.save_folder) if '.tfrecord' in f]) == 0

        pkl_fnames = []
        for folder in self._pkl_folders:
            folder = os.path.join(DATA_DIR, folder)

            pkl_fnames += [os.path.join(folder, f) for f in os.listdir(folder)
                                 if FileManager.train_rollouts_fname_suffix in f or \
                                    FileManager.eval_rollouts_fname_suffix in f]

        random.shuffle(pkl_fnames)

        tfrecord_fnames = []
        num_train = 0
        num_holdout = 0

        for pkl_fname in pkl_fnames:
            if os.path.splitext(FileManager.train_rollouts_fname_suffix)[0] in pkl_fname:
                tfrecord_fname = os.path.join(self._trav_graph.save_folder, 'train_{0:05d}.tfrecord'.format(num_train))
                num_train += 1
            elif os.path.splitext(FileManager.eval_rollouts_fname_suffix)[0] in pkl_fname:
                tfrecord_fname = os.path.join(self._trav_graph.save_folder, 'holdout_{0:05d}.tfrecord'.format(num_holdout))
                num_holdout += 1
            else:
                raise ValueError
            tfrecord_fnames.append(tfrecord_fname)

        pool = multiprocessing.Pool(num_processes)
        pool.starmap(RoadLabeller._write_tfrecord, zip(pkl_fnames,
                                                       tfrecord_fnames,
                                                       [self._image_shape]*len(pkl_fnames)))

    def _train_model(self):
        for step in range(self._train_steps):
            self._trav_graph.train_step()

            if step % self._eval_every_n_steps == 0:
                self._trav_graph.holdout_cost()

            if step > 0 and step % self._log_every_n_steps == 0:
                logger.record_tabular('step', step)
                self._trav_graph.log()
                logger.dump_tabular(print_func=logger.info)

            if step > 0 and step % self._save_every_n_steps == 0:
                self._trav_graph.save()

    def _eval_model(self):
        logger.info('Restoring model')
        self._trav_graph.restore()

        logger.info('Evaluating model')

        while True:
            obs, labels, probs, obs_holdout, labels_holdout, probs_holdout = self._trav_graph.eval()

            for obs_t, labels_t, probs_t, obs_holdout_t, labels_holdout_t, probs_holdout_t in \
                    zip(obs, labels, probs, obs_holdout, labels_holdout, probs_holdout):
                f, axes = plt.subplots(2, 4, figsize=(20, 5))

                axes[0, 0].imshow(obs_t)
                axes[0, 1].imshow(labels_t[...,0], cmap='Greys', vmin=0, vmax=1)
                axes[0, 2].imshow(probs_t[...,1], cmap='Greys', vmin=0, vmax=1)
                axes[0, 3].imshow(abs(labels_t[...,0] - probs_t[...,1]), cmap='Greys', vmin=0, vmax=1)

                axes[1, 0].imshow(obs_holdout_t)
                axes[1, 1].imshow(labels_holdout_t[...,0], cmap='Greys', vmin=0, vmax=1)
                axes[1, 2].imshow(probs_holdout_t[...,1], cmap='Greys', vmin=0, vmax=1)
                axes[1, 3].imshow(abs(labels_holdout_t[...,0] - probs_holdout_t[...,1]), cmap='Greys', vmin=0, vmax=1)

                plt.show()
