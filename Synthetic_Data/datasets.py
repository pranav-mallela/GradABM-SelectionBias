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

        # I think this this approach doesn't give you an explicit count of infections per time point but rather samples from a distribution across the entire timeline. 
        # prob_dist = rel_cols_df["Infectious_severe"][start:end]/rel_cols_df["Infectious_severe"][start:end].sum()
        # t = self.rng.choice(a=52, size=int(1e5), p=prob_dist)
           
        # # Sampling region
        # r = self.rng.choice(a=np.arange(self.num_regions), size=self.dataset_size)

        # # Sampling demographic
        # delta_r = self.rng.dirichlet(np.ones(self.num_demographics), size=self.num_regions) # matrix of demographic percentages by region
        # d = [self.rng.choice(a=np.arange(self.num_demographics), size=self.dataset_size, p=delta_r[region]) for region in r]
        ##########################################################################
        ##########################################################################
        # New Approach: Create population first and then sample infected from there and and also who is observed
        # We will only work with 1 region at this time
        ##########################################################################
        total_population = self.dataset_size
        
        # TODO: Each region will have its own demographic so we will do this by region
        demographic_distribution = [50, 20, 15, 10, 5]
        # Define the demographic categories
        demographic_labels = ['A', 'B', 'C', 'D', 'E']
        # Convert percentages to proportions
        demographic_proportions = np.array(demographic_distribution) / 100

        # Calculate the number of individuals in each demographic category
        demographic_counts = (demographic_proportions * total_population).astype(int)

        # Ensure the total matches N_total, if necessary, by adjusting the last group
        # This can help mitigate rounding issues
        demographic_counts[-1] = total_population - demographic_counts[:-1].sum()
        demographics = np.concatenate([np.full(count, label) for count, label in zip(demographic_counts, demographic_labels)])
        np.random.shuffle(demographics)
        
        df_individuals = pd.DataFrame({
            'id': np.arange(total_population),  # Unique identifier for each individual
            'demographic': demographics,  # Assigned demographics
            'infection_status': np.zeros(total_population, dtype=int)  # 0 = Not Infected, 1 = Infected
        })
         
        infection_rates = rel_cols_df["Infectious_severe"][start:end]
        normalized_rates = infection_rates/ infection_rates.sum() #these are probs now
        # We want the num of individuals to assign at each time point, t
        infected_per_timepoint = (normalized_rates * total_population).astype(int)

        # individuals = np.arange(total_population)
        # infected_individuals = []
        
        # # Sample individuals based on the number of infected per time point
        # for num_infected in infected_per_timepoint:
        #     infected_at_timepoint = np.random.choice(individuals, size=num_infected, replace=False)
        #     infected_individuals.append(infected_at_timepoint)
            
        for time_idx, num_infected in enumerate(infected_per_timepoint):
            # Randomly select individuals to infect at this time point
            infected_indices = np.random.choice(df_individuals.index, size=num_infected, replace=False)
            
            # Update their infection status
            df_individuals.loc[infected_indices, 'infection_status'] = 1
            
            # You can also store the time of infection if needed
            df_individuals.loc[infected_indices, 'infection_time'] = time_idx
            
        # model hospitalization rate for a dem d during week t 
        # # = number of hosp. / total pop. in region r
        # case 1: all demographics have the same weekly hospitalization rate
        # case 2: different rates per dem
        # pi_P = 0.01 * d
        # weekly_hosp_rate = expit(pi_P)
        
        #Let's explore case 1:
        weekly_hosp_rate = np.array([0.5, 0.5, 0.5 ,0.5, 0.5]) 

        # simulate a biased dataset using distribution Q
        pi_Q = expit(0.005 * d)

        # indicator variable
        obs = pi_Q > self.rng.uniform(size=self.dataset_size)

        df_individuals['obs'] = obs
        # X_total = np.stack((t, d, r), axis=-1)

        # return X_total, weekly_hosp_rate, obs, delta_r
        return df_individuals, df_individuals, weekly_hosp_rate
