"""
This module is how to setup a sample experiment.
"""
import numpy as np
from gym.spaces import Box

from experiments.base_experiment import *
from helper.CarlaHelper import update_config
import carla

SERVER_VIEW_CONFIG = {
}

SENSOR_CONFIG = {
    "CAMERA_X": 1280,
    "CAMERA_Y": 720,
}
OBSERVATION_CONFIG ={
    "CAMERA_OBSERVATION": False,
}

EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
    "Server_View": SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": SENSOR_CONFIG,
    "number_of_spawning_actors": 0,
}


class Experiment(BaseExperiment):
    def __init__(self):
        config=update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)
        super().__init__(config)


    def set_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

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

    def compute_reward(self, core, observation):
        """
        Reward function
        :return:
        :param core:
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

    def process_observation(self, core, observation):
        """
        Post processing of raw CARLA observations
        :param core: Core Environment
        :param observation: CARLA observations
        :return:
        """
        #self.set_server_view(core)

        post_observation = np.r_[core.normalize_coordinates(
            observation["location"].location.x,
            observation["location"].location.y)]
        return post_observation



    def set_server_view(self, core):

        """
        Apply server view to be in the sky between camera between start and end positions
        :param core:
        :return:
        """
        # spectator pointing to the sky to reduce rendering impact

        server_view_x = (
                self.experiment_config["Server_View"]["server_view_x_offset"]
                + (self.start_location.location.x + self.end_location.location.x) / 2
        )
        server_view_y = (
                self.experiment_config["Server_View"]["server_view_y_offset"]
                + (self.start_location.location.y + self.end_location.location.y) / 2
        )
        server_view_z = self.experiment_config["Server_View"]["server_view_height"]
        server_view_pitch = self.experiment_config["Server_View"]["server_view_pitch"]

        world = core.get_core_world()
        self.spectator = world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch),
            )
        )
