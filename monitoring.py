"""
monitoring.py
-------------
Utilities for tracking latent-space statistics during training.
Uses Welford's online algorithm for memory-efficient variance computation.
"""

import numpy as np
import matplotlib.pyplot as plt


class RunningVariance:
    """
    Welford's algorithm for computing sample variance incrementally.

    Parameters
    ----------
    n_dims : int
        Dimensionality of each incoming data vector.
    ddof : int
        Delta degrees of freedom (default 1).
    """

    def __init__(self, n_dims, ddof=1):
        self.ddof = ddof
        self.count = 0
        self.running_mean = np.zeros(n_dims)
        self.sum_sq_diff = np.zeros(n_dims)

    def update(self, observation):
        """Incorporate a single observation vector."""
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        self.sum_sq_diff += delta * (observation - self.running_mean)

    @property
    def variance(self):
        return self.sum_sq_diff / (self.count - self.ddof)

    @property
    def std_dev(self):
        return np.sqrt(self.variance)


class LatentVisualizer:
    """
    Tracks per-epoch latent-space variance and persists results to a file.

    Parameters
    ----------
    n_dims : int
        Number of latent dimensions to monitor.
    output_path : str
        File path where epoch-wise variance data is appended.
    """

    def __init__(self, n_dims, output_path: str):
        self.n_dims = n_dims
        self.output_path = output_path
        self._tracker = RunningVariance(n_dims=self.n_dims, ddof=0)
        self.epoch_variances = []

    def __call__(self, latent_vectors):
        """Shorthand for ``record_latent_batch``."""
        self.record_latent_batch(latent_vectors)

    def record_latent_batch(self, latent_vectors):
        """
        Feed a batch of latent vectors (numpy arrays) to the running estimator.
        """
        for vec in latent_vectors:
            self._tracker.update(vec)

    def render_heatmap(self):
        """
        Produce a heatmap of latent variance across recorded epochs
        and save to disk.
        """
        stacked = np.squeeze(np.array(self.epoch_variances)).T
        plt.figure()
        plt.imshow(stacked, cmap="hot", interpolation="nearest")
        plt.xlabel("Epoch")
        plt.ylabel("Latent Dimension")
        plt.colorbar()
        plt.savefig("latent_heatmap.png")
        plt.close()

    def finish_epoch(self):
        """
        Call at the end of every training epoch.
        Stores the variance snapshot and resets the accumulator.
        """
        self.epoch_variances.append([self._tracker.std_dev ** 2])

        with open(self.output_path, "a+") as fh:
            print(self.epoch_variances[-1][0], file=fh)

        self._tracker = RunningVariance(n_dims=self.n_dims, ddof=0)
