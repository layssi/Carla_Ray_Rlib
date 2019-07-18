import os
import random
import time
import sys
import cv2
import numpy as np

def add_carla_path(carla_path_config_file):
    carla_text_path = (os.path.dirname(os.path.realpath(__file__)) + "/" + carla_path_config_file)
    carla_path_file = open(carla_text_path, "r")
    carla_main_path = (carla_path_file.readline().split("\n"))[0]
    carla_path_file.close()
    carla_egg_file = (carla_main_path + "/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg")
    sys.path.append(os.path.expanduser(carla_egg_file))
    carla_python_interface = carla_main_path + "/PythonAPI/carla/"
    carla_server_binary = carla_main_path + "/CarlaUE4.sh"
    sys.path.append(os.path.expanduser(carla_python_interface))
    print(carla_python_interface)
    return carla_server_binary


def get_parent_dir(directory):
    return os.path.dirname(directory)


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :return: normalized image
    """
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.float32)



def spawn_vehicle_at(transform, vehicle_blueprint, world, autopilot=True, max_time=0.1):
    """
    Try to spawn a vehicle and give the vehicle time to be spawned and seen by the world before you say it is spawned

    :param transform: Location and Orientation of vehicle
    :param vehicle_blueprint: Vehicle Blueprint (We assign a random color)
    :param world: World
    :param autopilot: If True, AutoPilot is Enabled. If False, autopilot is disabled
    :param max_time: Maximum time in s to wait before you say that vehicle can not be spawned at current location
    :return: True if vehicle was added to world and False otherwise
    """

    # If the vehicle can not be spawned, it is OK
    previous_number_of_vehicles = len(world.get_actors().filter("*vehicle*"))

    # Get a random color
    color = random.choice(vehicle_blueprint.get_attribute("color").recommended_values)
    vehicle_blueprint.set_attribute("color", color)

    vehicle = world.try_spawn_actor(vehicle_blueprint, transform)

    wait_tick = 0.002  # Wait of 2ms to recheck if a vehicle is spawned
    if vehicle is not None:
        vehicle.set_autopilot(autopilot)
        # vehicle.set_simulate_physics(not(world.get_settings().synchronous_mode)) #Disable physics in synchronous mode
        world.tick()  # Tick the world so it creates the vehicle
        while previous_number_of_vehicles >= len(
            world.get_actors().filter("*vehicle*")
        ):
            time.sleep(wait_tick)  # Wait 2ms and check again
            max_time = max_time - wait_tick
            if max_time <= 0:  # Check for expiration time
                return False
        return vehicle
    return False


