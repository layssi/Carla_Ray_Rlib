from .BaseCarlaCore import BaseCarlaCore, CORE_CONFIG

class CarlaCore(BaseCarlaCore):
    def __init__(self, environment_config, experiment_config, core_config=None):
        """
        Initialize the server, clients, hero and sensors
        :param environment_config: Environment Configuration
        :param experiment_config: Experiment Configuration
        """
        self.core_config = core_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config
        super().init_server()