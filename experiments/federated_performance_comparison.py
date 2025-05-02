from data.federated_data import FederatedDataManager
import torch.nn as nn
from config import *
from experiments.experiment_logger import ExperimentLogger
from federated.server import FederatedServer


class FederatedPerformanceComparison:
    def __init__(self, configs: list[BaseConfig], save_model=False, save_results=False, plot_results=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        self.configs = configs

    def run(self, num_experiments=1):
        for _ in range(num_experiments):
            loss_fn = nn.CrossEntropyLoss()
            for c in self.configs:
                fed_config = FederatedConfig(c)
                loaders = FederatedDataManager(fed_config)
                client_loaders, test_loader = loaders.load_and_split_data()
                logger = ExperimentLogger.from_latest_run(c)
                if c.VERBOSE: print(f'Creating {fed_config.NAME} Model...')
                server = FederatedServer(fed_config, client_loaders, test_loader)
                server.train_environment(logger, loss_fn, save_results=self.save_results)
                server.test_results = server.evaluate_global_model(logger, loss_fn, save_results=self.save_results)
