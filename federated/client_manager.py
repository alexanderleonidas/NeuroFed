import copy
import random
from config import FederatedConfig
from models.trainable import Trainable
from training.train_single_model import train_single_model

class Client:
    def __init__(self, client_id, trainable: Trainable):
        self.client_id = client_id
        self.trainable = trainable

    def train_local_model(self, logger, global_model, loss_fn, save_results, communication_round):
        """Train the client's local model starting from the global model weights"""
        # Copy parameters from the global model to the client's local model
        global_state = copy.deepcopy(global_model.state_dict())

        # Handle the state dict mapping for DP models
        if hasattr(self.trainable.model, "_module"):
            # This is a DP wrapped model
            mapped_state = {}
            for key, value in global_state.items():
                # Map keys from the standard model to DP model format
                mapped_state[f"_module.{key}"] = value
            self.trainable.model.load_state_dict(mapped_state)
        else:
            # Standard model
            self.trainable.model.load_state_dict(global_state)

        # Training loop
        train_single_model(logger, self.trainable, loss_fn, save_results=save_results, client_id=self.client_id, communication_round=communication_round)

        # Return model updates and number of samples - ensure a consistent format
        if hasattr(self.trainable.model, "_module"):
            # Convert DP model state dict back to standard format for aggregation
            dp_state = self.trainable.model.state_dict()
            standard_state = {}
            for key, value in dp_state.items():
                if key.startswith("_module."):
                    standard_state[key[8:]] = value
            return standard_state, len(self.trainable.train_loader.dataset)
        else:
            return self.trainable.model.state_dict(), len(self.trainable.train_loader.dataset)


class ClientManager:
    def __init__(self, clients: list[Client], config:FederatedConfig):
        self.clients = clients
        self.config = config
        self.selected_clients = []

    def select_clients(self):
        """Randomly select a fraction of clients for training in this round"""
        num_clients_to_select = max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        self.selected_clients = random.sample(self.clients, num_clients_to_select)
        return self.selected_clients

    def train_selected_clients(self, logger, global_model, loss_fn, save_results, communication_round):
        """Train all selected clients and return their updates"""
        client_updates = []
        client_sizes = []

        for client in self.selected_clients:
            if self.config.VERBOSE: print(f'--------------- Training client with ID {client.client_id} ---------------')
            update, size = client.train_local_model(logger, global_model, loss_fn, save_results, communication_round)
            client_updates.append(update)
            client_sizes.append(size)

        return client_updates, client_sizes


