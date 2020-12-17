from tensorflow import keras
import keras.backend as K

import numpy as np
from scipy.stats import norm


class InverseGaussianWeightedLoss:
    ''' Compute the inverse weighting of a Gaussian Distribution.
    '''

    def __init__(self, samples, min_weight=1e-3, max_weight=1000):
        self.mu, self.std = norm.fit(samples)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_loss(self, y_true, y_pred):
        std = self.std * K.ones_like(y_true)
        mu  = self.mu  * K.ones_like(y_true)
        pi  = np.pi    * K.ones_like(y_true)

        weights = ((std*K.sqrt(2.0*pi)) *
                   K.exp(0.5*K.square((y_true-mu)/std)))
        loss = K.square(y_true-y_pred)*weights
        return loss

    def __str__(self):
        return ("Inverse Gaussian Weighted Loss with " +
                f"mu={self.mu} and std={self.std}")
