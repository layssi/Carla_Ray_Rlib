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

# For helper submodules to inherit "imports", the imporst  have to be before importing helper functions
import os
import signal
import ray
from ray import tune
from carla_env import CarlaEnv
from helper.list_procs import search_procs_by_name
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from ray.tune import grid_search, run_experiments

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
    #register_carla_model()
    ray.init()
    run_experiments({
        "carla-a3c": {
            "run": "A3C",
            "env": CarlaEnv,
            "stop":{"episodes_total":30000000},#"training_iteration":5000000},
            "checkpoint_at_end":True,
            "checkpoint_freq":1,
            "config": {
                "env_config": env_config,
                "num_gpus_per_worker": 0,
                "num_cpus_per_worker":2,
                "num_workers": 2,
                "gamma": 0.99,  # random.choice([0.5, 0.8, 0.9, 0.95, 0.99]),

            },
        },
    },
    resume= False,
    )
#        kill_server()
