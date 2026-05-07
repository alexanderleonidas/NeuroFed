import torch
from models.trainable import Trainable
from training.evaluate_model import evaluate_model
from .client_manager import ClientManager
from config import FederatedConfig


class FederatedServer:
    def __init__(self, config: FederatedConfig, client_loaders, test_loader):
        self.config = config
        self.global_model = Trainable(self.config, test_loader=test_loader, global_fed_model=True)
        self.client_manager = ClientManager(client_loaders, config)
        self.aggregator = FedAvg()


    def train_environment(self, logger, loss_fn, save_results, save_model):
        """Run the federated learning process"""
        if self.config.VERBOSE:
            eps = "Epsilon" if hasattr(self.config, 'EPSILON') else ""
            print("Round", "Client", "Epoch", "Loss", "Accuracy (%)", "Precision", "Recall", "F1 Score", "Time (s)", "CPU (%)","", eps, sep="\t")
            print(100 * "-")

        global_state_before_training = self.global_model.model.state_dict().copy()
        for round_num in range(self.config.COMMUNICATION_ROUNDS):
            # Select clients for this round
            self.client_manager.select_clients()

            # Train selected clients
            client_updates, client_sizes = self.client_manager.train_selected_clients(logger, global_state_before_training, loss_fn, round_num, save_results, save_model)

            # Aggregate updates
            aggregated_state = self.aggregator.aggregate(global_state_before_training, client_updates, client_sizes)
            # Update the global model with the new state
            self.global_model.model.load_state_dict(aggregated_state)
            # The global state dict for the *next* round is the one just calculated
            global_state_before_training = self.global_model.model.state_dict().copy()
            if save_model: logger.save_model(self.global_model.model)

    def evaluate_global_model(self, logger, loss_fn, save_results):
        # Evaluate global model
        if self.config.VERBOSE: print(f'------------ Testing global model ------------')
        test_results = evaluate_model(logger, self.global_model, loss_fn, self.global_model.test_loader, save_results=save_results)
        return test_results



class FedAvg:
    """Federated Averaging aggregation strategy"""
    @staticmethod
    def aggregate(global_state_before_training, client_deltas, client_sizes):
        """
        Aggregate client updates using weighted averaging based on dataset sizes

        :param global_state_before_training: The global model state before aggregation.
        :type global_state_before_training: dict
        :param client_deltas: The model deltas of the clients after training.
        :type client_deltas: list
        :param client_sizes: The size of each client.
        :type client_sizes: list
        """
        if not client_deltas:
            print("Warning: No client updates received. Global model remains unchanged.")
            return global_state_before_training

        total_size = sum(client_sizes)

        # Initialize a dictionary to hold the aggregated updates
        aggregated_delta_dict = {}
        for key in client_deltas[0].keys():
            aggregated_delta_dict[key] = torch.zeros_like(client_deltas[0][key]).to(client_deltas[0][key].device)

        # First, calculate raw weights
        weights = [size / total_size for size in client_sizes]

        for i, delta_dict in enumerate(client_deltas):
            for key in aggregated_delta_dict.keys():
                # Add weighted delta to the aggregated delta dict
                aggregated_delta_dict[key] += delta_dict[key] * weights[i]

        # Calculate the new global model state: w^{t+1} = w^t + aggregated_delta
        new_global_state = {}
        for key in global_state_before_training.keys():
            if key in aggregated_delta_dict:
                # Add the aggregated delta to the corresponding parameter in the original global state
                new_global_state[key] = global_state_before_training[key] + aggregated_delta_dict[key]
            else:
                # If a parameter didn't exist in client updates (unlikely in standard models), keep the original
                new_global_state[key] = global_state_before_training[key]
                print(f"Warning: Key '{key}' not found in aggregated delta. Using original value.")

        # The aggregate method should return the new state dict
        return new_global_state
