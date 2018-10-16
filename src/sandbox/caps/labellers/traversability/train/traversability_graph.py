import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

from gcg.data.logger import logger
from gcg.tf import tf_utils
from gcg.misc import utils

from sandbox.caps.labellers.traversability.train import fcn

class TraversabilityGraph:

    def __init__(self, image_shape, save_folder, **kwargs):
        ### environment
        self._image_shape = image_shape

        ### saving
        self._save_folder = save_folder

        ### training
        self._batch_size = kwargs['batch_size']
        self._num_classes = 2
        self._inference_only = kwargs.get('inference_only', False)
        self._lr = kwargs.get('lr')
        self._weight_decay = kwargs.get('weight_decay')
        self._gpu_device = kwargs.get('gpu_device')
        self._gpu_frac = kwargs.get('gpu_frac')

        ### setup the model
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

        self._log_stats = defaultdict(list)

    ##################
    ### Properties ###
    ##################

    @property
    def save_folder(self):
        return self._save_folder

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_input_output_placeholders(self):
        obs_shape = list(self._image_shape)
        obs_dtype = tf.uint8

        with tf.variable_scope('input_output_placeholders'):
            tf_obs_ph = tf.placeholder(obs_dtype, [None] + obs_shape, name='tf_obs_ph')
            tf_labels_ph = tf.placeholder(obs_dtype, [None] + obs_shape[:-1], name='tf_labels_ph')

        return tf_obs_ph, tf_labels_ph

    def _graph_input_output_tfrecords(self, train_or_holdout,
                                      shuffle_buffer_size=1000,
                                      prefetch_buffer_size_multiplier=4):
        assert train_or_holdout == 'train' or train_or_holdout == 'holdout'

        def parse(ex):
            features = {"image": tf.FixedLenFeature((), tf.string),
                        "label": tf.FixedLenFeature((), tf.string)}
            parsed_features = tf.parse_single_example(ex, features)

            image = tf.decode_raw(parsed_features['image'], tf.uint8)
            label = tf.decode_raw(parsed_features['label'], tf.uint8)

            image = tf.reshape(image, self._image_shape)
            label = tf.reshape(label, self._image_shape[:2] + [1])

            return image, label

        tfrecord_fnames = [os.path.join(self._save_folder, fname) for fname in os.listdir(self._save_folder)
                           if os.path.splitext(fname)[1] == '.tfrecord' and train_or_holdout in fname]
        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(parse)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer_size))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size_multiplier * self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset)
        tf_image_tr, tf_label_tr = iterator.get_next()

        return dataset_init_op, tf_image_tr, tf_label_tr

    def _graph_inference(self, tf_obs_ph):
        ### resize
        tf_obs_resized = tf_obs_ph
        if tf_obs_resized.get_shape()[-1].value == 1:
            tf_obs_resized = tf.tile(tf_obs_resized, (1, 1, 1, 3))
        elif tf_obs_resized.get_shape()[-1].value == 3:
            pass
        else:
            raise ValueError
        tf_obs_float = tf.cast(tf_obs_resized, tf.float32)

        ### create network
        network = fcn.FCN16VGG()
        network.build(tf_obs_float, train=not self._inference_only, num_classes=self._num_classes)

        ### get relevant outputs
        tf_scores = network.upscore32
        # tf_probs = network.prob32
        tf_probs = tf.nn.sigmoid(tf_scores[..., 1])

        return tf_scores, tf_probs

    def _graph_cost(self, tf_labels_ph, tf_scores):
        num_classes = self._num_classes

        logits = tf.reshape(tf_scores, (-1, num_classes))

        # epsilon = tf.constant(value=1e-4)
        # tf_labels = tf.cast(tf.expand_dims(tf_labels_ph, 3), tf.int32)
        # tf_labels = tf.cast(tf.clip_by_value(tf_labels, 0, num_classes - 1), tf.uint8)
        # labels = tf.to_float(tf.reshape(tf.one_hot(tf_labels, num_classes, axis=3), (-1, num_classes)))
        # softmax = tf.nn.softmax(logits) + epsilon

        assert self._num_classes == 2
        logits = logits[..., 1]
        labels = tf.to_float(tf.reshape(tf_labels_ph, (-1,)))

        # cross_entropy = -tf.reduce_sum(
        #     labels * tf.log(softmax), reduction_indices=[1]) # TODO: change to built in softmax with logits

        assert self._num_classes == 2
        pos_weight = tf.reduce_sum(1 - labels) / tf.reduce_max([tf.reduce_sum(labels), 1.])
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits,
                                                                 targets=labels,
                                                                 pos_weight=pos_weight)

        tf_mse = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        tf_accuracy = tf.reduce_mean(tf.to_float(
            tf.logical_or(tf.logical_and(logits > 0, labels > 0.5),
                          tf.logical_and(logits < 0, labels < 0.5))))

        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            tf_weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            tf_weight_decay = 0
        tf_cost = tf_mse + tf_weight_decay

        return tf_cost, tf_mse, tf_accuracy

    def _graph_optimize(self, tf_cost, tf_policy_vars):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-4)
            gradients = optimizer.compute_gradients(tf_cost, var_list=tf_policy_vars)
            tf_opt = optimizer.apply_gradients(gradients)
        return tf_opt

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = tf_utils.create_session_and_graph(gpu_device=self._gpu_device,
                                                                  gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            ### create input output placeholders
            tf_obs_ph, tf_labels_ph = self._graph_input_output_placeholders()

            ### inference
            policy_scope = 'traversability'
            with tf.variable_scope(policy_scope):
                tf_scores, tf_probs = self._graph_inference(tf_obs_ph)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope=policy_scope), key=lambda v: v.name)

            if not self._inference_only:
                ### cost and optimize
                train_init_op, tf_obs_tr, tf_labels_tr = self._graph_input_output_tfrecords('train')
                with tf.variable_scope(policy_scope, reuse=True):
                    tf_scores_tr, tf_probs_tr = self._graph_inference(tf_obs_tr)
                tf_cost_tr, tf_mse_tr, tf_acc_tr = self._graph_cost(tf_labels_tr, tf_scores_tr)
                tf_opt_tr = self._graph_optimize(tf_cost_tr, tf_trainable_policy_vars)

                holdout_init_op, tf_obs_tr_holdout, tf_labels_tr_holdout = self._graph_input_output_tfrecords('holdout')
                with tf.variable_scope(policy_scope, reuse=True):
                    tf_scores_tr_holdout, tf_probs_tr_holdout = self._graph_inference(tf_obs_tr_holdout)
                tf_cost_tr_holdout, tf_mse_tr_holdout, tf_acc_tr_holdout = self._graph_cost(tf_labels_tr_holdout, tf_scores_tr_holdout)
            else:
                tf_obs_tr = tf_labels_tr = tf_scores_tr = tf_probs_tr = tf_cost_tr = tf_mse_tr = tf_acc_tr = tf_opt_tr = \
                    tf_obs_tr_holdout = tf_labels_tr_holdout = tf_scores_tr_holdout = tf_probs_tr_holdout = \
                    tf_cost_tr_holdout = tf_acc_tr_holdout = None

            ### savers
            tf_saver = tf.train.Saver(tf_policy_vars, max_to_keep=None)

            ### initialize
            tf_sess.run([tf.global_variables_initializer()])
            if not self._inference_only:
                tf_sess.run([train_init_op, holdout_init_op])

        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_ph': tf_obs_ph,
            'labels_ph': tf_labels_ph,
            'scores': tf_scores,
            'probs': tf_probs,

            'obs_tr': tf_obs_tr,
            'labels_tr': tf_labels_tr,
            'scores_tr': tf_scores_tr,
            'probs_tr': tf_probs_tr,
            'cost': tf_cost_tr,
            'mse': tf_mse_tr,
            'acc': tf_acc_tr,
            'opt': tf_opt_tr,

            'obs_tr_holdout': tf_obs_tr_holdout,
            'labels_tr_holdout': tf_labels_tr_holdout,
            'scores_tr_holdout': tf_scores_tr_holdout,
            'probs_tr_holdout': tf_probs_tr_holdout,
            'cost_holdout': tf_cost_tr_holdout,
            'acc_holdout': tf_acc_tr_holdout,
            'saver': tf_saver,
            'policy_vars': tf_policy_vars
        }

    ################
    ### Training ###
    ################

    def train_step(self):
        cost, mse, acc, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                       self._tf_dict['mse'],
                                                       self._tf_dict['acc'],
                                                       self._tf_dict['opt']],
                                                      feed_dict=None)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
        self._log_stats['acc'].append(acc)

    def holdout_cost(self):
        cost, acc = self._tf_dict['sess'].run([self._tf_dict['cost_holdout'],
                                               self._tf_dict['acc_holdout']])
        assert(np.isfinite(cost))

        self._log_stats['Cost holdout'].append(cost)
        self._log_stats['acc holdout'].append(acc)

    #################
    ### Debugging ###
    #################

    def eval(self):
        obs, labels, probs, obs_holdout, labels_holdout, probs_holdout = \
            self._tf_dict['sess'].run([self._tf_dict['obs_tr'],
                                       self._tf_dict['labels_tr'],
                                       self._tf_dict['probs_tr'],
                                       self._tf_dict['obs_tr_holdout'],
                                       self._tf_dict['labels_tr_holdout'],
                                       self._tf_dict['probs_tr_holdout']],
                                      feed_dict=None)

        return obs, labels, probs, obs_holdout, labels_holdout, probs_holdout

    #####################
    ### Model methods ###
    #####################

    def get_model_outputs(self, observations):
        observations = np.asarray(observations)
        assert observations.shape[-1] == self._image_shape[-1], 'do not have same number of channels'
        if tuple(observations.shape[1:3]) != tuple(self._image_shape[:2]):
            observations = np.array([utils.imresize(o_t, self._image_shape) for o_t in observations])

        feed_dict = {
            self._tf_dict['obs_ph']: observations
        }

        probs,  = self._tf_dict['sess'].run([self._tf_dict['probs']], feed_dict=feed_dict)

        return probs

    ######################
    ### Saving/loading ###
    ######################

    @property
    def _ckpt_fname(self):
        return os.path.join(self._save_folder, 'checkpoint.ckpt')

    @property
    def is_ckpt(self):
        return os.path.exists(os.path.splitext(self._ckpt_fname)[0])

    def save(self):
        self._tf_dict['saver'].save(self._tf_dict['sess'], self._ckpt_fname, write_meta_graph=False)

    def restore(self):
        self._tf_dict['saver'].restore(self._tf_dict['sess'], self._ckpt_fname)

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()
