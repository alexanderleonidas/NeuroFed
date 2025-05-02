import torch
from models.trainable import Trainable
from training.evaluate_model import evaluate_model
from .client_manager import ClientManager, Client
from config import FederatedConfig


class FederatedServer:
    def __init__(self, config: FederatedConfig, client_loaders, test_loader):
        self.test_loader = test_loader
        self.config = config
        self.global_model = Trainable(self.config, global_fed_model=True)
        self.client_manager = self.__set_client_manager(client_loaders)
        self.aggregator = FedAvg()

    def __set_client_manager(self, client_loaders):
        clients = []
        for idx, (client_train, client_val) in enumerate(client_loaders):
            clients.append(Client(idx, Trainable(self.config, client_train, client_val)))
        return ClientManager(clients, self.config)

    def train_environment(self, logger, loss_fn, save_results):
        """Run the federated learning process"""
        test_results = None
        for round_num in range(self.config.COMMUNICATION_ROUNDS):
            if self.config.VERBOSE:
                print(f"\n========== Communication Round {round_num + 1}/{self.config.COMMUNICATION_ROUNDS} ==========")

            # Select clients for this round
            selected_clients = self.client_manager.select_clients()
            if self.config.VERBOSE:
                print(f"Selected {len(selected_clients)} clients for training")

            # Train selected clients
            client_updates, client_sizes = self.client_manager.train_selected_clients(
                logger, self.global_model.model, loss_fn, save_results=save_results, communication_round=round_num)

            # Aggregate updates
            self.global_model.model = self.aggregator.aggregate(
                self.global_model.model, client_updates, client_sizes)

    def evaluate_global_model(self, logger, loss_fn, save_results):
        # Evaluate global model
        if self.config.VERBOSE: print(f'------------ Testing global model ------------')
        test_results = evaluate_model(logger, self.global_model, loss_fn, self.test_loader, save_results=save_results)
        return test_results



class FedAvg:
    """Federated Averaging aggregation strategy"""
    @staticmethod
    def aggregate(global_model, client_updates, client_sizes):
        """Aggregate client updates using weighted averaging based on dataset sizes"""
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)

        # Initialize a dictionary to hold the aggregated updates
        aggregated_dict = {}
        for key in global_dict.keys():
            aggregated_dict[key] = torch.zeros_like(global_dict[key])

        # First calculate raw weights
        raw_weights = [size / total_size for size in client_sizes]
        # Then normalize to ensure they sum to 1
        weight_sum = sum(raw_weights)
        normalized_weights = [w / weight_sum for w in raw_weights]

        # Use the normalized weights in the aggregation loop
        for i, client_dict in enumerate(client_updates):
            for key in global_dict.keys():
                aggregated_dict[key] += client_dict[key] * normalized_weights[i]

        # Update the global model
        global_model.load_state_dict(aggregated_dict)
        return global_model