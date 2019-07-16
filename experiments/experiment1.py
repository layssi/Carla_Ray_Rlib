"""
This module is how to setup a sample experiment.
"""
import numpy as np
from gym.spaces import Box

from experiments.base_experiment import BaseExperiment

OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": True,
    "COLLISION_OBSERVATION": True,
    "LOCATION_OBSERVATION": True,
}

EXPERIMENT_CONFIG = {
    "number_of_spawning_actors": 10,
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
}


class Experiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.experiment_config.update(EXPERIMENT_CONFIG)
        self.experiment_config.update(OBSERVATION_CONFIG)

    def set_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def post_process_observation(self, core, observation):
        """
        Post processing of raw CARLA observations
        :param core: Core Environment
        :param observation: CARLA observations
        :return:
        """
        post_observation = np.r_[core.normalize_coordinates(
            observation["location"].location.x,
            observation["location"].location.y)]
        return post_observation

    def initialize_reward(self, core):
        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        # Get vehicle start location so reward can be calculated as total distance traveled
        self.start_pos_normalized_x, self.start_pos_normalized_y = core.normalize_coordinates(
            self.start_location.location.x,
            self.start_location.location.y)
        self.previous_distance = None

    def compute_reward(self, observation):
        """
        Reward function
        :return:
        :param observation:
        :return:
        """
        # Get vehicle start location so reward can be calculated as total distance traveled
        normalized_x = observation[0]
        normalized_y = observation[1]

        distance_reward = float(
            np.linalg.norm(
                [
                    normalized_x - self.start_pos_normalized_x,
                    normalized_y - self.start_pos_normalized_y,
                ])
        )

        if self.previous_distance is None:
            reward = 0
        else:
            reward = distance_reward - self.previous_distance

        self.previous_distance = distance_reward

        return reward
