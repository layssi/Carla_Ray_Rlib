"""A3C Algorithm. Tested yet with Carla.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import ray
from ray import tune
from carla_env import CarlaEnv
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from ray.tune import grid_search, run_experiments
from helper.CarlaHelper import kill_server

ENV_CONFIG = {"RAY": True, # Are we running an experiment in Ray
              "DEBUG_MODE": False,
              "Experiment": "experiment3",
              }

env_config = ENV_CONFIG.copy()
env_config.update(
    {
        "RAY": True,  # Are we running an experiment in Ray
        "DEBUG_MODE": False,
    }
)


if __name__ == "__main__":
    kill_server()
    ray.init()
    run_experiments({
        "carla-a3c": {
            "run": "A3C",
            "env": CarlaEnv,
            "stop": {"episodes_total":30000000}, #"training_iteration":5000000},
            "checkpoint_at_end": True,
            "checkpoint_freq": 1,
            "config": {
                "env_config": env_config,
                "num_gpus_per_worker": 0.25,
                "num_cpus_per_worker": 2,
                "num_workers": 3,
                "gamma": 0.99,  # random.choice([0.5, 0.8, 0.9, 0.95, 0.99]),

            },
        },
    },
    resume= False,
    )
