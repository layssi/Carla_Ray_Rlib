from experiments.base_experiment import *
from helper.CarlaHelper import spawn_vehicle_at, post_process_image, update_config
import random
import numpy as np
from gym.spaces import  Box
from itertools import cycle

SERVER_VIEW_CONFIG = {
}

SENSOR_CONFIG = {
    "CAMERA_NORMALIZED": True,
    "FRAMESTACK": 4,
}
OBSERVATION_CONFIG ={
    "CAMERA_OBSERVATION": True,
}

EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
    "Server_View": SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": SENSOR_CONFIG,
    "number_of_spawning_actors": 10000,
    "hero_vehicle_model": "vehicle.mini.cooperst",
}

ENV_CONFIG = {"RAY": True, "DEBUG_MODE": False}  # Are we running an experiment in Ray


class Experiment(BaseExperiment):
    def __init__(self):
        config=update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)
        super().__init__(config)


        self.max_actors = 30
        self.randomized_vehicle_spawn_point = None

        self.environment_config = ENV_CONFIG.copy()

        self.environment_config.update(
            {
                "RAY": True,  # Are we running an experiment in Ray
                "DEBUG_MODE": False,
                "corridor_length": 5,
            }
        )

    def initialize_reward(self, core):
        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        self.previous_distance = 0
        self.base_x = 0
        self.base_y = 0

        self.frame_stack = 4  # can be 1,2,3,4
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.start_location = None


    def set_observation_space(self):
        num_of_channels = 1
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.experiment_config["SENSOR_CONFIG"]["CAMERA_X"],
                self.experiment_config["SENSOR_CONFIG"]["CAMERA_Y"],
                num_of_channels * self.experiment_config["SENSOR_CONFIG"]["FRAMESTACK"],
            ),
            dtype=np.float32,
        )
        self.observation_space = image_space

    def process_observation(self, core, observation):
        """
        Process observations according to your experiment
        :param core:
        :param observation:
        :return:
        """
        self.set_server_view(core)
        image = post_process_image(observation['camera'],
                                   normalized = self.experiment_config["SENSOR_CONFIG"]["CAMERA_NORMALIZED"],
                                   grayscale = self.experiment_config["SENSOR_CONFIG"]["CAMERA_GRAYSCALE"]
        )
        image = image[:, :, np.newaxis]

        if self.prev_image_0 is None:  # can be improved
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1
        #ToDO Hamid. Fix the images stack
        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, image], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        # uncomment to save the observations (Normalized must be False)
        '''
        cv2.imwrite('./input_img0.jpg', image)
        cv2.imwrite('./input_img1.jpg', self.prev_image_0)
        cv2.imwrite('./input_img2.jpg', self.prev_image_1)
        cv2.imwrite('./input_img3.jpg', self.prev_image_2)
        '''
        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image

        return images


    def compute_reward(self, core, observation):
        """
        Reward function
        :param observation:
        :return:
        """
        c = float(np.sqrt(np.square(self.hero.get_location().x - self.base_x) + \
                          np.square(self.hero.get_location().y - self.base_y)))
        if c > self.previous_distance + 1e-2:
            reward = 1
        else:
            reward = 0

        # print("\n", self.previous_distance)
        self.previous_distance = c
        # print("\n", c)

        if c > 50:
            # print("\n", self.base_x, hero.get_location().x,
            #      "\n", self.base_y, hero.get_location().y)
            self.base_x = self.hero.get_location().x
            self.base_y = self.hero.get_location().y
            print("Reached the milestone!")
            self.previous_distance = 0
        return reward

    def spawn_actors(self, core):
        # Get a list of all the vehicle blueprints
        world = core.get_core_world()
        vehicle_blueprints = world.get_blueprint_library().filter("vehicle.*")
        car_blueprints = [
            x
            for x in vehicle_blueprints
            if int(x.get_attribute("number_of_wheels")) == 4
        ]
        # Get all spawn Points
        spawn_points = list(world.get_map().get_spawn_points())

        # Now we are ready to spawn all the vehicles (except the hero)
        count = 0  # self.experiment_config["number_of_spawning_actors"]

        self.randomized_vehicle_spawn_point = spawn_points.copy()

        while count > 1:
            random.shuffle(self.randomized_vehicle_spawn_point, random.random)
            next_spawn_point = self.randomized_vehicle_spawn_point[count]

            # Try to spawn but if you can't, just move on
            next_vehicle = spawn_vehicle_at(
                next_spawn_point,
                random.choice(car_blueprints),
                world,
                autopilot=True,
                max_time=0.1,
            )
            print(count)
            if next_vehicle is not False:
                self.spawn_point_list.append(next_spawn_point)
                self.vehicle_list.append(next_vehicle)
                count -= 1
        if len(self.vehicle_list) > self.max_actors:
            for v in self.vehicle_list:  # do we need this?
                v.destroy()
            self.vehicle_list = []
            self.spawn_point_list = []

        # print(world.get_actors().filter("vehicle.*"))
        print('number of actors: ', len(self.vehicle_list))

        # spawn hero
        # self.spawn_hero()
        # return self.hero

    def spawn_hero(self, core, transform, autopilot=False):
        world = core.get_core_world()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        random.shuffle(self.randomized_vehicle_spawn_point, random.random)
        next_spawn_point = self.randomized_vehicle_spawn_point[0]
        if next_spawn_point in self.spawn_point_list:
            random.shuffle(self.randomized_vehicle_spawn_point, random.random)
            next_spawn_point = self.randomized_vehicle_spawn_point[0]
        # spawn hero
        # Hamid what's the dif between spawn_vehicle_at and try_spawn_actor
        super().spawn_hero(core, next_spawn_point, autopilot=False)

        print("Hero spawned!")
        self.base_x = self.hero.get_location().x
        self.base_y = self.hero.get_location().y
        self.start_location = next_spawn_point