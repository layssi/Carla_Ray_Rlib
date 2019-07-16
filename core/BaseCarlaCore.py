import math
import os
import random
import signal
import subprocess
import time

import carla
import numpy as np

from helper.CameraManager import CameraManager
from helper.CarlaDebug import get_actor_display_name
from helper.CollisionManager import CollisionManager
from helper.list_procs import search_procs_by_name

"""
Generic Carla colors (Not being used but can be useful)
"""
RED = carla.Color(255, 0, 0)
GREEN = carla.Color(0, 255, 0)
BLUE = carla.Color(47, 210, 231)
CYAN = carla.Color(0, 255, 255)
YELLOW = carla.Color(255, 255, 0)
ORANGE = carla.Color(255, 162, 0)
WHITE = carla.Color(255, 255, 255)

CORE_CONFIG = {
    "RAY_DELAY": 3,  # Delay between 0 & RAY_DELAY before starting server so not all servers are launched simultaneously
    "RETRIES_ON_ERROR": 30,
    "timeout": 2.0,
    "host": "localhost",
    "map_buffer": 1.2,  # To find the minimum and maximum coordinates of the map
               }


class BaseCarlaCore:
    def __init__(self, environment_config, experiment_config, core_config=None):
        """
        Initialize the server, clients, hero and sensors
        :param environment_config: Environment Configuration
        :param experiment_config: Experiment Configuration
        """
        if core_config is None:
            core_config = CORE_CONFIG

        self.core_config = core_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config

        self.init_server(self.core_config["RAY_DELAY"])

        self.client, self.world, self.town_map, self.actors = self.__connect_client(
            self.core_config["host"],
            self.server_port,
            self.core_config["timeout"],
            self.core_config["RETRIES_ON_ERROR"],
            self.experiment_config["Disable_Rendering_Mode"],
            True,
            self.experiment_config["Weather"],
        )
        self.set_map_dimensions()
        self.hero = None
        self.camera_manager = None
        self.collision_manager = None

    # ==============================================================================
    # -- ServerSetup -----------------------------------------------------------
    # ==============================================================================
    def init_server(self, ray_delay=0):
        """
        Start a server on a random port
        :param ray_delay: Delay so not all servers start simultaneously causing race condition
        :return:
        """
        if self.environment_config["RAY"] is False:
            try:
                # Kill all PIDs that start with Carla. Do this if you running a single server or before an experiment
                for pid, name in search_procs_by_name("Carla").items():
                    os.kill(pid, signal.SIGKILL)
            except:
                pass

        # Generate a random port to connect to. You need one port for each server-client
        if self.environment_config["DEBUG_MODE"]:
            self.server_port = 2000
        else:
            self.server_port = random.randint(10000, 60000)

        # Create a new server process and start the client.
        if self.environment_config["RAY"] is True:
            # Ray tends to start all processes simultaneously. This causes problems
            # => random delay to start individual servers
            delay_sleep = random.uniform(0, ray_delay)
            time.sleep(delay_sleep)

        if self.environment_config["DEBUG_MODE"] is True:
            # Big Screen for Debugging
            self.experiment_config["SENSOR_CONFIG"]["CAMERA_X"] = 900
            self.experiment_config["SENSOR_CONFIG"]["CAMERA_Y"] = 1200
            self.experiment_config["quality_level"] = "High"

        # Run the server process
        server_command = [
            self.environment_config["SERVER_BINARY"],
            self.experiment_config["server_map"],
            "-windowed",
            "-ResX={}".format(self.experiment_config["SENSOR_CONFIG"]["CAMERA_X"]),
            "-ResY={}".format(self.experiment_config["SENSOR_CONFIG"]["CAMERA_Y"]),
            "-carla-server",
            "-carla-port={}".format(self.server_port),
            "-fps={}".format(self.experiment_config["fps"]),  #
            "-quality-level =",
            self.experiment_config["quality_level"],
            "--no-rendering",
            "-carla-server-timeout = 10000ms",
        ]
        if not self.experiment_config["Disable_Rendering_Mode"]:
            server_command.remove("--no-rendering")

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    # ==============================================================================
    # -- ClientSetup -----------------------------------------------------------
    # ==============================================================================

    @staticmethod
    def __connect_client(host, port, timeout, num_retries, disable_rendering_mode, sync_mode, weather):
        """
        Connect the client

        :param host: The host servers
        :param port: The server port to connect to
        :param timeout: The server takes time to get going, so wait a "timeout" and re-connect
        :param num_retries: Number of times to try before giving up
        :param disable_rendering_mode: True to disable rendering
        :param sync_mode: True for RL
        :param weather: The weather to start the world
        :return:
        """
        for i in range(num_retries):
            try:
                carla_client = carla.Client(host, port)
                carla_client.set_timeout(timeout)
                world = carla_client.get_world()

                settings = world.get_settings()
                settings.no_rendering_mode = disable_rendering_mode
                world.apply_settings(settings)

                settings = world.get_settings()
                settings.synchronous_mode = sync_mode
                settings.fixed_delta_seconds = 1/20

                world.apply_settings(settings)

                town_map = world.get_map()
                actors = world.get_actors()
                world.set_weather(weather)

                return carla_client, world, town_map, actors

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, num_retries))
                time.sleep(1)
        # if (i + 1) == num_retries:
        raise Exception("Trouble is brewing. Can not connect to server. Try increasing timeouts or num_retries")

    # ==============================================================================
    # -- ClientSetup -----------------------------------------------------------
    # ==============================================================================

    def apply_server_view(self, server_view_x, server_view_y, server_view_z, server_view_pitch):
        """
        :param server_view_x: X location of server camera
        :param server_view_y: Y location of server camera
        :param server_view_z: Z location of server camera
        :param server_view_pitch: Pitch of server camera
        :return: None
        """
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch),
            )
        )

    # ==============================================================================
    # -- MapDigestionsSetup -----------------------------------------------------------
    # ==============================================================================

    def set_map_dimensions(self):
        """
        From the spawn points, we get min and max and add some buffer so we can normalize the location of agents (0..1)
        This allows you to get the location of the vehicle between 0 and 1

        :input
        self.core_config["map_buffer"]. Because we use spawn points, we add a buffer as vehicle can drive off the road

        :output:
        self.coord_normalization["map_normalization"] = Using larger of (X,Y) axis to normalize x,y
        self.coord_normalization["map_min_x"] = minimum x coordinate
        self.coord_normalization["map_min_y"] = minimum y coordinate
        :return: None
        """

        map_buffer = self.core_config["map_buffer"]
        spawn_points = list(self.world.get_map().get_spawn_points())

        min_x = min_y = 1000000
        max_x = max_y = -1000000

        for spawn_point in spawn_points:
            min_x = min(min_x, spawn_point.location.x)
            max_x = max(max_x, spawn_point.location.x)

            min_y = min(min_y, spawn_point.location.y)
            max_y = max(max_y, spawn_point.location.y)

        center_x = (max_x+min_x)/2
        center_y = (max_y+min_y)/2

        x_buffer = (max_x - center_x) * map_buffer
        y_buffer = (max_y - center_y) * map_buffer

        min_x = center_x - x_buffer
        max_x = center_x + x_buffer

        min_y = center_y - y_buffer
        max_y = center_y + y_buffer

        self.coord_normalization = {"map_normalization": max(max_x - min_x, max_y - min_y),
                                    "map_min_x": min_x,
                                    "map_min_y": min_y}

    def normalize_coordinates(self, input_x, input_y):
        """
        :param input_x: X location of your actor
        :param input_y: Y location of your actor
        :return: The normalized location of your actor
        """
        output_x = (input_x - self.coord_normalization["map_min_x"]) / self.coord_normalization["map_normalization"]
        output_y = (input_y - self.coord_normalization["map_min_y"]) / self.coord_normalization["map_normalization"]

        # ToDO Possible bug (Clipped the observation and still didn't stop the observations from being under
        output_x = float(np.clip(output_x, 0, 1))
        output_y = float(np.clip(output_y, 0, 1))

        return output_x, output_y

    # ==============================================================================
    # -- SensorSetup -----------------------------------------------------------
    # ==============================================================================

    def setup_sensors(self, experiment_config, hero, synchronous_mode=True):

        """
        This function sets up hero vehicle sensors

        :param experiment_config: Sensor configuration for you sensors
        :param hero: Hero vehicle
        :param synchronous_mode: set to True for RL
        :return:
        """

        if experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]:
            self.camera_manager = CameraManager(
                hero,
                experiment_config["SENSOR_CONFIG"]["CAMERA_X"],
                experiment_config["SENSOR_CONFIG"]["CAMERA_Y"],
                experiment_config["SENSOR_CONFIG"]["CAMERA_FOV"],
            )
            sensor = experiment_config["SENSOR_CONFIG"]["SENSOR"].value
            self.camera_manager.set_sensor(sensor, synchronous_mode=synchronous_mode)
            transform_index = experiment_config["SENSOR_CONFIG"][
                "SENSOR_TRANSFORM"
            ].value
        if experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            self.collision_manager = CollisionManager(
                hero, synchronous_mode=synchronous_mode
            )

    def reset_sensors(self, experiment_config):
        """
        Destroys sensors that were setup in this class
        :param experiment_config: sensors configured in the experiment
        :return:
        """
        if experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]:
            if self.camera_manager is not None:
                self.camera_manager.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            if self.collision_manager is not None:
                self.collision_manager.destroy_sensor()

    # ==============================================================================
    # -- CameraSensor -----------------------------------------------------------
    # ==============================================================================

    def record_camera(self, record_state):
        self.camera_manager.set_recording(record_state)

    def render_camera_lidar(self, render_state):
        self.camera_manager.set_rendering(render_state)

    def update_camera(self):
        self.camera_manager.read_image_queue()

    def get_camera_data(self, normalized=True):
        return self.camera_manager.get_camera_data(normalized)

    # ==============================================================================
    # -- CollisionSensor -----------------------------------------------------------
    # ==============================================================================

    def update_collision(self):
        self.collision_manager.read_collision_queue()

    def get_collision_data(self):
        return self.collision_manager.get_collision_data()

    # ==============================================================================
    # -- Hero -----------------------------------------------------------
    # ==============================================================================
    def spawn_hero(self, transform, vehicle_blueprint, world, autopilot=False):
        """
        This function spawns the hero vehicle. It makes sure that if a hero exists=>destroy the hero and respawn

        :param transform: Hero location
        :param vehicle_blueprint: Hero vehicle blueprint
        :param world: World
        :param autopilot: Autopilot Status
        :return:
        """
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        hero_car_blueprint = world.get_blueprint_library().find(vehicle_blueprint)
        hero_car_blueprint.set_attribute("role_name", "hero")

        while self.hero is None:
            self.hero = world.try_spawn_actor(hero_car_blueprint, transform)

        self.hero.set_autopilot(autopilot)

    def get_hero(self):

        """
        Get hero vehicle
        :return:
        """
        return self.hero

    # ==============================================================================
    # -- OtherForNow -----------------------------------------------------------
    # ==============================================================================

    def get_core_world(self):
        return self.world

    def kill_server(self):
        # Kill all PIDs that start with Carla. Do this if you running a single server
        for pid, name in search_procs_by_name("Carla").items():
            os.kill(pid, signal.SIGKILL)

    def get_nearby_vehicles(self, world, hero_actor, max_distance=200):
        vehicles = world.get_actors().filter("vehicle.*")
        surrounding_vehicles = []
        surrounding_vehicle_actors = []
        _info_text = []
        if len(vehicles) > 1:
            _info_text += ["Nearby vehicles:"]
            for x in vehicles:
                if x.id != hero_actor:
                    loc1 = hero_actor.get_location()
                    loc2 = x.get_location()
                    distance = math.sqrt(
                        (loc1.x - loc2.x) ** 2
                        + (loc1.y - loc2.y) ** 2
                        + (loc1.z - loc2.z) ** 2
                    )
                    vehicle = {}
                    if distance < max_distance:
                        vehicle["vehicle_type"] = get_actor_display_name(x, truncate=22)
                        vehicle["vehicle_location"] = x.get_location()
                        vehicle["vehicle_velocity"] = x.get_velocity()
                        vehicle["vehicle_distance"] = distance
                        surrounding_vehicles.append(vehicle)
                        surrounding_vehicle_actors.append(x)
