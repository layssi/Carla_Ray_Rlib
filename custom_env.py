"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# For helper submodules to inherit "imports", the import  have to be before importing helper functions
import os
import signal
import ray
from ray import tune
from carla_env import CarlaEnv
from helper.list_procs import search_procs_by_name
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog


ENV_CONFIG = {"RAY": True, "DEBUG_MODE": False}  # Are we running an experiment in Ray

env_config = ENV_CONFIG.copy()
env_config.update(
    {
        "RAY": True,  # Are we running an experiment in Ray
        "DEBUG_MODE": False,
    }
)


def kill_server():
    # Kill all PIDs that start with Carla. Do this if you running a single server
    for pid, name in search_procs_by_name("Carla").items():
        os.kill(pid, signal.SIGKILL)


class CustomModel(Model):

    """Example of a custom model.

    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        print(input_dict)
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(
            input_dict, self.obs_space, self.action_space, num_outputs, options
        )
        return self.fcnet.outputs, self.fcnet.last_layer


if __name__ == "__main__":
 #   try:
        #    tune.register_env("CarlaEnv", lambda _: CarlaEnv)
    kill_server()

    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
    "A3C", #"PPO"
    stop={"timesteps_total": 1000000},
    checkpoint_freq=1,
    config={
        "env": CarlaEnv,  # CarlaEnv,SimpleCorridor,  # or "corridor" if registered above
        "model": {"custom_model": "my_model"},
#        "lr": 1e-2,  # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 4,  # parallelism
        "num_gpus_per_worker": 0.2,
        "env_config": env_config,
    },
        resume=False,
    )
#    except:
#        kill_server()
