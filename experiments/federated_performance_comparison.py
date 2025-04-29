from data.federated_data import FederatedDataManager
import torch.nn as nn
from config import *
from federated.server import FederatedServer
from utils import save_model_safely


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
                print(fed_config.EXPERIMENT_TYPE)
                if c.VERBOSE: print(f'Creating {fed_config.NAME} Model...')
                server = FederatedServer(fed_config, client_loaders, test_loader)
                server.train_environment(loss_fn, save_results=self.save_results)
                server.test_results = server.evaluate_global_model(loss_fn, save_results=self.save_results)

                if self.save_model:
                    for client in server.client_manager.clients:
                        if fed_config.VERBOSE and save_model_safely(client.trainable.model, fed_config, client.client_id):
                            print(f'Client {client.client_id} model saved in {fed_config.SAVED_MODELS_PATH}')
                    if fed_config.VERBOSE and save_model_safely(server.global_model.model, fed_config):
                        print(f'Global model saved in {fed_config.SAVED_MODELS_PATH}')
