"""Example of a custom model with CARLA.  This currently works with experiment one (1 Dimensional Observation)

This example shows:
  - using a custom environment
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.

 THIS EXPERIMENT HAS NOT BEEN FULLY TESTED

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray import tune
from carla_env import CarlaEnv
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from ray.tune import run_experiments, grid_search
from helper.CarlaHelper import kill_server



ENV_CONFIG = {"RAY": True, "DEBUG_MODE": False}  # Are we running an experiment in Ray

env_config = ENV_CONFIG.copy()
env_config.update(
    {
        "RAY": True,  # Are we running an experiment in Ray
        "DEBUG_MODE": False,
        "Experiment": "experiment1",
    }
)





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
    print("THIS EXPERIMENT HAS NOT BEEN FULLY TESTED")
    kill_server()
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
    "PPO",
    stop={"timesteps_total": 1000000},
    checkpoint_freq=1,
    config={
        "env": CarlaEnv,  # CarlaEnv,SimpleCorridor,  # or "corridor" if registered above
        "model": {"custom_model": "my_model"},
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 4,  # parallelism
        "num_gpus_per_worker": 0.2,
        "env_config": env_config,
    },
        resume=False,
    )
