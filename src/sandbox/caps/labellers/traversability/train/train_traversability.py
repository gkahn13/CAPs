import os, glob
import random
import itertools

import numpy as np
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from matplotlib.path import Path

from gcg.data import mypickle
from gcg.misc import utils
from gcg.data.logger import logger
from sandbox.caps.labellers.traversability.train.traversability_graph import TraversabilityGraph

def extract_images_from_pkls(pkl_folder, save_folder, maxsaved, image_shape, rescale, bordersize):
    """
    :param pkl_folder: folder containing pkls with training images
    :param save_folder: where to save the resulting images
    :param maxsaved: how many images to save
    :param image_shape: shape of the image
    :param rescale: make rescale times bigger, for ease of labelling
    :param bordersize: how much to pad with 0s, for ease of labelling
    """
    random.seed(0)

    fnames = glob.glob(os.path.join(pkl_folder, '*.pkl'))
    random.shuffle(fnames)
    logger.info('{0} files to read'.format(len(fnames)))
    fnames = itertools.cycle(fnames)

    im_num = 0
    while im_num < maxsaved:
        fname = next(fnames)
        rollout = random.choice(mypickle.load(fname)['rollouts'])
        obs = random.choice(rollout['observations_im'])

        height, width, channels = image_shape

        im = np.reshape(obs, image_shape)
        im = utils.imresize(im, (rescale * height, rescale * width, channels))
        im = np.pad(im, ((bordersize, bordersize), (bordersize, bordersize), (0, 0)), 'constant')
        if im.shape[-1] == 1:
            im = im[:, :, 0]
        Image.fromarray(im).save(os.path.join(save_folder, 'image_{0:06d}.jpg'.format(im_num)))
        im_num += 1

    logger.info('Saved {0} images'.format(im_num))

def create_labels(save_folder, image_shape, rescale, bordersize):
    """
    :param save_folder: where images are saved
    :param image_shape: shape of the image
    :param rescale: make rescale times bigger, for ease of labelling
    :param bordersize: how much to pad with 0s, for ease of labelling
    """

    ### load csv
    csv_fname = os.path.join(save_folder, 'via_region_data.csv')
    assert (os.path.exists(csv_fname))
    csv = pandas.read_csv(csv_fname)

    ### image indices
    xdim = int(rescale) * image_shape[0] + 2 * bordersize
    ydim = int(rescale) * image_shape[1] + 2 * bordersize
    x, y = np.meshgrid(np.arange(xdim), np.arange(ydim))
    indices = np.vstack((y.flatten(), x.flatten())).T

    ### create labels
    num_labelled = 0
    for i in range(len(csv)):
        fname = csv['#filename'][i]
        region_shape_attrs = eval(csv['region_shape_attributes'][i])
        if 'name' not in region_shape_attrs.keys():
            continue

        assert (region_shape_attrs['name'] == 'polygon')
        xy = np.stack((region_shape_attrs['all_points_x'], region_shape_attrs['all_points_y'])).T
        label = 1 - Path(xy).contains_points(indices).reshape(ydim, xdim).T  # 0 is no collision
        label = label.astype(np.uint8)

        label_fname = os.path.join(save_folder, 'label_' + os.path.splitext(fname)[0] + '.jpg')
        Image.fromarray(label).save(label_fname)

        num_labelled += 1

    logger.info('{0} were labelled'.format(num_labelled))

def create_training_data(save_folder, image_shape, rescale, bordersize, holdout_pct):
    """
    :param save_folder: where images are saved
    :param image_shape: shape of the image
    :param rescale: make rescale times bigger, for ease of labelling
    :param bordersize: how much to pad with 0s, for ease of labelling
    :param holdout_pct: how much data to holdout
    """

    ### read image, label pairs
    label_fnames = glob.glob(os.path.join(save_folder, 'label*'))
    random.shuffle(label_fnames)

    height, width, channels = image_shape

    images_train, labels_train = [], []
    images_holdout, labels_holdout = [], []
    for i, label_fname in enumerate(label_fnames):
        image_fname = label_fname.replace('label_', '')

        image = np.asarray(Image.open(image_fname))
        label = np.asarray(Image.open(label_fname))

        # reduce image back down
        image = image[bordersize:-bordersize, bordersize:-bordersize]
        image = utils.imresize(image, (height, width, channels))
        assert (tuple(image.shape) == tuple(image_shape))

        # reduce label back down
        label = label[bordersize:-bordersize, bordersize:-bordersize]
        label = utils.imresize(label, (height, width, 1), Image.BILINEAR)
        label = label[:, :, 0]
        label = (label > 0.5)
        assert (tuple(label.shape) == (height, width))

        if i / float(len(label_fnames)) > holdout_pct:
            images_train.append(image)
            labels_train.append(label)
        else:
            images_holdout.append(image)
            labels_holdout.append(label)

    np.save(os.path.join(save_folder, 'data_train_images.npy'), np.array(images_train))
    np.save(os.path.join(save_folder, 'data_train_labels.npy'), np.array(labels_train))

    np.save(os.path.join(save_folder, 'data_holdout_images.npy'), np.array(images_holdout))
    np.save(os.path.join(save_folder, 'data_holdout_labels.npy'), np.array(labels_holdout))

    logger.info('Saved train and holdout')

def train_traversability_graph(save_folder, image_shape, trav_graph_params):
    ### create graph
    trav_graph = TraversabilityGraph(image_shape, save_folder, **trav_graph_params)

    ### load data
    images = np.load(os.path.join(save_folder, 'data_train_images.npy'))
    labels = np.load(os.path.join(save_folder, 'data_train_labels.npy'))
    images_holdout = np.load(os.path.join(save_folder, 'data_holdout_images.npy'))
    labels_holdout = np.load(os.path.join(save_folder, 'data_holdout_labels.npy'))

    train_steps = int(trav_graph_params['train_steps'])
    eval_every_n_steps = int(trav_graph_params['eval_every_n_steps'])
    batch_size = trav_graph_params['batch_size']
    log_every_n_steps = int(trav_graph_params['log_every_n_steps'])
    save_every_n_steps = int(trav_graph_params['save_every_n_steps'])

    for step in range(train_steps):
        indices = np.random.randint(low=0, high=len(images), size=batch_size)
        images_batch = images[indices]
        labels_batch = labels[indices]

        trav_graph.train_step(images_batch, labels_batch)

        if step % eval_every_n_steps == 0:
            indices = np.random.randint(low=0, high=len(images_holdout), size=batch_size)
            images_holdout_batch = images_holdout[indices]
            labels_holdout_batch = labels_holdout[indices]
            trav_graph.holdout_cost(images_holdout_batch, labels_holdout_batch)

        if step > 0 and step % log_every_n_steps == 0:
            trav_graph.log()
            logger.dump_tabular(print_func=logger.info)

        if step > 0 and step % save_every_n_steps == 0:
            trav_graph.save()

def eval_traversability_graph(save_folder, image_shape, trav_graph_params):
    ### create graph
    trav_graph = TraversabilityGraph(image_shape, save_folder, **trav_graph_params)

    ### load data
    images = np.load(os.path.join(save_folder, 'data_train_images.npy'))
    labels = np.load(os.path.join(save_folder, 'data_train_labels.npy'))
    images_holdout = np.load(os.path.join(save_folder, 'data_holdout_images.npy'))
    labels_holdout = np.load(os.path.join(save_folder, 'data_holdout_labels.npy'))

    ### restore policy
    trav_graph.restore()

    eval_folder = os.path.join(save_folder, 'eval')
    os.makedirs(eval_folder, exist_ok=True)

    def save(name, observation, label):
        label = label.astype(float)

        label_pred = trav_graph.get_model_outputs([observation])[0]
        label_pred = label_pred[:, :, 1]

        f, axes = plt.subplots(1, 4, figsize=(20, 5))

        is_color = (observation.shape[-1] == 3)
        if is_color:
            axes[0].imshow(observation)
        else:
            axes[0].imshow(observation[:, :, 0], cmap='Greys_r')
        axes[1].imshow(label, cmap='Greys', vmin=0, vmax=1)
        axes[2].imshow(label_pred, cmap='Greys', vmin=0, vmax=1)
        axes[3].imshow(abs(label - label_pred), cmap='Greys', vmin=0, vmax=1)

        axes[0].set_title('Image')
        axes[1].set_title('Ground truth')
        axes[2].set_title('Prediction')
        axes[3].set_title('Prediction error')

        f.savefig(os.path.join(eval_folder, name), dpi=100)
        plt.close(f)

    for i, (image, label) in enumerate(zip(images, labels)):
        save('train_{0:04d}.png'.format(i), image, label)

    for i, (image, label) in enumerate(zip(images_holdout, labels_holdout)):
        save('holdout_{0:04d}.png'.format(i), image, label)