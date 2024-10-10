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
        Data: One row represents deidentified patient data
        Columns: time (in weeks), demographic, region, weekly_hosp_rate, observed (True if sampled, else False)

        """
        # Sampling time
        weekly_df = pd.read_csv("./hybridODE_weekly_data.csv")
        rel_cols_df = weekly_df[["time", "beta", "Ca", "Infectious_mild", "Infectious_severe", "Hospitalized_recovered", "Hospitalized_deceased"]]

        start = 0
        timesteps = 52
        end = start + timesteps

        prob_dist = rel_cols_df["Infectious_severe"][start:end]/rel_cols_df["Infectious_severe"][start:end].sum()
        t = self.rng.choice(a=52, size=int(1e5), p=prob_dist)

        # Sampling region
        r = self.rng.choice(a=np.arange(self.num_regions), size=self.dataset_size)

        # Sampling demographic
        delta_r = self.rng.dirichlet(np.ones(self.num_demographics), size=self.num_regions) # matrix of demographic percentages by region
        d = [self.rng.choice(a=np.arange(self.num_demographics), size=self.dataset_size, p=delta_r[region]) for region in r]

        # model hospitalization rate for a dem d during week t 
        # # = number of hosp. / total pop. in region r
        # case 1: all demographics have the same weekly hospitalization rate
        # case 2: different rates per dem
        pi_P = 0.01 * d
        weekly_hosp_rate = expit(pi_P)

        # simulate a biased dataset using distribution Q
        pi_Q = expit(0.005 * d)

        # indicator variable
        obs = pi_Q > self.rng.uniform(size=self.dataset_size)

        X_total = np.stack((t, d, r), axis=-1)

        return X_total, weekly_hosp_rate, obs, delta_r
