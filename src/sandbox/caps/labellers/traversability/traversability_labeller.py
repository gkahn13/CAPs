import numpy as np

from gcg.labellers.labeller import Labeller
from sandbox.caps.labellers.traversability.train.traversability_graph import TraversabilityGraph

class TraversabilityLabeller(Labeller):

    def __init__(self, env_spec, tf_sess, **kwargs):
        self._env_spec = env_spec

        with tf_sess.as_default():
            self._trav_graph = TraversabilityGraph(image_shape=self._env_spec.observation_im_space.shape,
                                                   **kwargs)
            self._trav_graph.restore()

    def label(self, observations, goals):
        ### get traversability prediction
        observations_im, _ = observations
        pred_labels = self._trav_graph.get_model_outputs(observations_im)

        ### 1 for traversable, 0 for not
        trav_area = 1 - pred_labels[:, :, :, 1]

        ### compute center of traversable area (in width direction)
        height, width = trav_area.shape[1:]
        xpos, _ = np.meshgrid(np.arange(width), np.arange(height))
        centers = np.array([xpos[ta > 0.5].mean() for ta in trav_area])
        centers = (centers - width/2.) / (width/2.)
        assert (np.all(centers >= -1.) and np.all(centers <= 1.))

        goals = centers

        # import matplotlib.pyplot as plt
        # f, axes = plt.subplots(1, 2)
        # axes[0].imshow(observations_im[0][:,:,0], cmap='Greys_r')
        # axes[1].imshow(trav_area[0], cmap='Greys_r')
        # axes[1].set_title('center: {0:.2f}'.format(goals[0]))
        # plt.show(block=False)
        # plt.pause(0.1)
        # import IPython; IPython.embed()
        # plt.close(f)

        return goals