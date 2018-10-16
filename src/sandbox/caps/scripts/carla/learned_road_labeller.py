import argparse
import os, random
import multiprocessing
import PIL

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gcg.data.logger import logger
from gcg.data.file_manager import FileManager, DATA_DIR
from gcg.data import mypickle
from gcg.misc import utils

from sandbox.caps.labellers.traversability.train.traversability_graph import TraversabilityGraph
from sandbox.caps.envs.carla.carla_coll_speed_road_env import CarlaCollSpeedRoadEnv


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(pkl_fname, tfrecord_fname, obs_shape):
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
                        'image': _bytes_feature(image.tostring()),
                        'label': _bytes_feature(rightlane_mask.tostring()),
                    }))
            writer.write(example.SerializeToString())

    writer.close()

class CollSpeedLearnedroad(object):

    def __init__(self, save_folder, pkl_folders, obs_shape=(50, 100, 3)):
        self._save_folder = os.path.join(DATA_DIR, save_folder)
        os.makedirs(self._save_folder, exist_ok=True)

        logger.setup(display_name=save_folder,
                     log_path=os.path.join(self._save_folder, 'log.txt'),
                     lvl='debug')

        self._pkl_fnames = []
        for folder in pkl_folders:
            folder = os.path.join(DATA_DIR, folder)

            self._pkl_fnames += [os.path.join(folder, f) for f in os.listdir(folder)
                                 if FileManager.train_rollouts_fname_suffix in f or \
                                    FileManager.eval_rollouts_fname_suffix in f]

        random.shuffle(self._pkl_fnames)

        self._obs_shape = list(obs_shape)

    def create_training_data(self, num_processes=10):
        assert len([f for f in os.listdir(self._save_folder) if '.tfrecord' in f]) == 0

        tfrecord_fnames = []
        num_train = 0
        num_holdout = 0

        for pkl_fname in self._pkl_fnames:
            if os.path.splitext(FileManager.train_rollouts_fname_suffix)[0] in pkl_fname:
                tfrecord_fname = os.path.join(self._save_folder, 'train_{0:05d}.tfrecord'.format(num_train))
                num_train += 1
            elif os.path.splitext(FileManager.eval_rollouts_fname_suffix)[0] in pkl_fname:
                tfrecord_fname = os.path.join(self._save_folder, 'holdout_{0:05d}.tfrecord'.format(num_holdout))
                num_holdout += 1
            else:
                raise ValueError
            tfrecord_fnames.append(tfrecord_fname)

        pool = multiprocessing.Pool(num_processes)
        pool.starmap(write_tfrecord, zip(self._pkl_fnames, tfrecord_fnames, [self._obs_shape]*len(self._pkl_fnames)))

    def train_model(self):
        ### create graph
        trav_graph = TraversabilityGraph(self._obs_shape, self._save_folder, **labeller_params)

        train_steps = int(labeller_params['train_steps'])
        eval_every_n_steps = int(labeller_params['eval_every_n_steps'])
        log_every_n_steps = int(labeller_params['log_every_n_steps'])
        save_every_n_steps = int(labeller_params['save_every_n_steps'])

        for step in range(train_steps):
            trav_graph.train_step()

            if step % eval_every_n_steps == 0:
                trav_graph.holdout_cost()

            if step > 0 and step % log_every_n_steps == 0:
                logger.record_tabular('step', step)
                trav_graph.log()
                logger.dump_tabular(print_func=logger.info)

            if step > 0 and step % save_every_n_steps == 0:
                trav_graph.save()

    def eval_model(self):
        logger.info('Creating model')
        trav_graph = TraversabilityGraph(self._obs_shape, self._save_folder, **labeller_params)

        logger.info('Restoring model')
        trav_graph.restore()

        logger.info('Evaluating model')

        while True:
            obs, labels, probs, obs_holdout, labels_holdout, probs_holdout = trav_graph.eval()

            for obs_t, labels_t, probs_t, obs_holdout_t, labels_holdout_t, probs_holdout_t in \
                    zip(obs, labels, probs, obs_holdout, labels_holdout, probs_holdout):
                f, axes = plt.subplots(2, 4, figsize=(20, 5))

                axes[0, 0].imshow(obs_t)
                axes[0, 1].imshow(labels_t[...,0], cmap='Greys', vmin=0, vmax=1)
                axes[0, 2].imshow(probs_t, cmap='Greys', vmin=0, vmax=1)
                axes[0, 3].imshow(abs(labels_t[...,0] - probs_t), cmap='Greys', vmin=0, vmax=1)

                axes[1, 0].imshow(obs_holdout_t)
                axes[1, 1].imshow(labels_holdout_t[...,0], cmap='Greys', vmin=0, vmax=1)
                axes[1, 2].imshow(probs_holdout_t, cmap='Greys', vmin=0, vmax=1)
                axes[1, 3].imshow(abs(labels_holdout_t[...,0] - probs_holdout_t), cmap='Greys', vmin=0, vmax=1)

                axes[0, 0].set_title('Input')
                axes[0, 1].set_title('Ground truth segmentation')
                axes[0, 2].set_title('Predicted segmentation')
                axes[0, 3].set_title('abs(Ground truth - predicted)')

                axes[0, 0].set_ylabel('Training')
                axes[1, 0].set_ylabel('Holdout')

                plt.show(block=False)
                plt.pause(0.1)
                response = input('Press enter to continue, or "quit" to exit\n')
                if response == 'quit':
                    return
                plt.close(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str, choices=('train', 'eval'))
    parser.add_argument('--save_folder', type=str, default='caps/carla/learned_road_labeller')
    parser.add_argument('--pkl_folders', nargs='+', default=['caps/carla/coll_speed/dql'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    labeller_params = {
        'train_steps': int(1e6),
        'eval_every_n_steps': 5,
        'batch_size': 32,
        'log_every_n_steps': int(1e1),
        'save_every_n_steps': int(1e2),
        'lr': 1.e-4,
        'weight_decay': 1e-4,
        'gpu_device': args.gpu,
        'gpu_frac': 0.8,
    }

    model = CollSpeedLearnedroad(args.save_folder, args.pkl_folders)
    if args.run == 'train':
        model.create_training_data()
        model.train_model()
    elif args.run == 'eval':
        model.eval_model()
    else:
        raise NotImplementedError
