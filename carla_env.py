"""
This is a sample carla environment. It does basic functionality.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Put import that you want "helper" modules to import
import time
import gym
from gym.utils import seeding
from helper.CarlaHelper import add_carla_path

ENV_CONFIG = {
    "RAY": True,  # True if you are  running an experiment in Ray
    "DEBUG_MODE": False,
    "CARLA_PATH_CONFIG_FILE": "CARLA_PATH.txt",  # IN this file, put the path to your CARLA FOLDER
}

CARLA_SERVER_BINARY = add_carla_path(ENV_CONFIG["CARLA_PATH_CONFIG_FILE"])
ENV_CONFIG.update({"SERVER_BINARY": CARLA_SERVER_BINARY})

import carla

#Choose your expreimet and Core
from experiments.experiment1 import Experiment
from core.CarlaCore1 import CarlaCore

from helper.CarlaDebug import draw_spawn_points, get_actor_display_name, \
    split_actors, get_actor_status, print_spawn_point


class CarlaEnv(gym.Env):
    def __init__(self, config=None):
        if config is None:
            config = ENV_CONFIG
        self.environment_config = config
        carla_server_binary = add_carla_path(ENV_CONFIG["CARLA_PATH_CONFIG_FILE"])
        self.environment_config.update({"SERVER_BINARY": carla_server_binary})

        self.seed()

        self.experiment = Experiment()
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.experiment_config = self.experiment.get_experiment_config()

        self.core = CarlaCore(self.environment_config, self.experiment_config)
        self.experiment.spawn_actors(self.core)
        self.experiment.initialize_reward(self.core)
        self.reset()

    def reset(self):
        self.core.reset_sensors(self.experiment_config)
        self.experiment.spawn_hero(self.core, self.experiment.start_location, autopilot=False)

        self.core.setup_sensors(
            self.experiment.experiment_config,
            self.experiment.get_hero(),
            self.core.get_core_world().get_settings().synchronous_mode,
        )
        self.experiment.initialize_reward(self.core)
        self.experiment.set_server_view(self.core)
        self.experiment.experiment_tick(self.core, action=None)
        obs, info = self.experiment.get_observation(self.core)
        obs = self.experiment.process_observation(self.core, obs)
        return obs

    def step(self, action):
        # assert action in [0, 13], action
        self.experiment.experiment_tick(self.core, action)
        observation, info = self.experiment.get_observation(self.core)
        observation = self.experiment.process_observation(self.core, observation)
        reward = self.experiment.compute_reward(self.core,observation)
        done = self.experiment.get_done_status()
        return observation, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":

    env = CarlaEnv()
    obs = env.reset()
    try:
        if env.environment_config["DEBUG_MODE"]:
            dir(carla)
            dir(carla.LaneInvasionEvent)
            dir(carla.LaneInvasionEvent.crossed_lane_markings)

            world = env.core.get_core_world()
            hero = env.experiment.hero

            draw_spawn_points(world)
            print_spawn_point(world)
            position, velocity, control, heading = get_actor_status(hero)
            print(
                "Position:",
                [position.location.x, position.location.y],
                " Velocity:",
                [velocity.x, velocity.y, velocity.z],
                " Heading:",
                heading,
            )
            print(get_actor_display_name(hero))
            # print_blueprint_attributes(world.get_blueprint_library())

            Vehicles, Traffic_lights, Speed_limits, walkers = split_actors(
                world.get_actors()
            )
            print("vehicles", Vehicles)
            env.core.get_nearby_vehicles(world, hero, max_distance=200)

        for _ in range(100):
            obs = env.reset()
            if env.environment_config["DEBUG_MODE"]:
                if env.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]:
                    env.core.record_camera(True)
                    env.core.render_camera_lidar(True)

            done = False
            while done is False:
                t = time.time()
                observation, reward, done, info = env.step(1)  # Forward
                # obs, reward, done, info = env2.step(2)  # Forward
                # print ("observation:",observation," Reward::{:0.2f}".format(reward * 1000))

                elapsed = time.time() - t
            # print("Elapsed (ms):{:0.2f}".format(elapsed * 1000))

    except (KeyboardInterrupt, SystemExit):
        env.core.kill_server()
