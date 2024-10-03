import numpy as np
import pandas as pd
from typing import Tuple
from scipy.special import expit  # Logistic function to constrain values between (0, 1)

class SyntheticDataGenerator():
    def __init__(self, dataset_size: int, num_regions: int, num_demographics: int) -> None:
        self.dataset_size = dataset_size
        self.num_regions = num_regions
        self.num_demographics = num_demographics
        self.rng = np.random.default_rng()

    def simulate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO: write interpretation of data structure.
        
        """
        t = self.rng.choice(a=np.arange(1, 53), size=self.dataset_size)
        d = self.rng.choice(a=np.arange(self.num_demographics), size=self.dataset_size)
        # use delta_r to choose d
        # use time to sample individual points
        r = self.rng.choice(a=np.arange(self.num_regions), size=self.dataset_size)

        # vector of demographic percentages by region
        delta_r = self.rng.dirichlet(np.ones(self.num_demographics), size=self.num_regions)

        # linear combination of factors influencing hospitalization rate
        pi_P = 0.01 * t + 0.01 * d + 0.01 * r
        hospitalization_rate = expit(pi_P)

        # simulate a biased dataset using distribution Q
        pi_Q = expit(0.005 * t + 0.005 * d + 0.005 * r) # ranges from 0 to 0.005 * (t_max + d_max + r_max)

        # indicator variable
        obs = pi_Q > self.rng.uniform(size=self.dataset_size)

        X_total = np.stack((t, d, r), axis=-1)

        return X_total, hospitalization_rate, obs, delta_r
