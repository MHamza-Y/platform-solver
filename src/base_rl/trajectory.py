import pandas as pd
import numpy as np


class Trajectory:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.rewards = []
        self.actions = []

    def add_sample(self, obs, next_obs, reward, action):
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_dataframe(self, ep_id=None):
        time_steps = range(len(self.observations))
        df = pd.DataFrame.from_dict(self.__dict__)
        df["ep_id"] = ep_id
        df["time_step"] = time_steps
        return df
