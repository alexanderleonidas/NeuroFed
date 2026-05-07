import random
from config import FederatedConfig
from models.trainable import Trainable
from training.train_single_model import train_single_model

class Client:
    def __init__(self, client_id, trainable: Trainable):
        self.client_id = client_id
        self.trainable = trainable


    def train_local_model(self, logger, global_state_dict, loss_fn, communication_round, save_results, save_model):
        """Train the client's local model starting from the global model weights"""

        initial_global_state = {key: value.clone() for key, value in global_state_dict.items()}

        # Handle the state dict mapping for DP models
        if hasattr(self.trainable.model, "_module"):
            # This is a DP wrapped model
            mapped_state = {}
            for key, value in initial_global_state.items():
                # Map keys from the standard model to DP model format
                mapped_state[f"_module.{key}"] = value
            try:
                self.trainable.model.load_state_dict(mapped_state)
            except Exception as e:
                print(f"Client {self.client_id}: Error loading mapped state dict: {e}")
                raise
        else:
            # Standard model
            try:
                self.trainable.model.load_state_dict(initial_global_state)
            except Exception as e:
                print(f"Client {self.client_id}: Error loading state dict: {e}")
                raise

        # Training loop
        train_single_model(logger, self.trainable, loss_fn, save_results=save_results, save_model=save_model, client_id=self.client_id, communication_round=communication_round)

        local_state_after_training = self.trainable.model.state_dict().copy()
        # Return model updates and number of samples - ensure a consistent format
        if hasattr(self.trainable.model, "_module"):
            # Convert DP model state dict back to standard format for aggregation
            standard_state = {}
            for key, value in local_state_after_training.items():
                if key.startswith("_module."):
                    standard_state[key[8:]] = value
                else:
                    # Should not happen if _module exists, but for safety
                    standard_state[key] = value
        else:
            standard_state = local_state_after_training

        delta_state = {}
        for key in initial_global_state.keys():  # Use keys from the initial global state as reference
            if key in standard_state:
                # Calculate the difference
                delta_state[key] = standard_state[key] - initial_global_state[key]
            else:
                # This case should ideally not happen if model architectures are consistent
                print(f"Client {self.client_id}: Warning: Key '{key}' not found in local state after training.")
                raise

        # Return the delta state dict and the number of samples
        return delta_state, len(self.trainable.train_loader.dataset)


class ClientManager:
    def __init__(self, client_loaders, config:FederatedConfig):
        self.config = config
        self.clients = self.__set_client_manager(client_loaders)
        self.selected_clients = []

    def __set_client_manager(self, client_loaders):
        clients = []
        for idx, (client_train, client_val) in enumerate(client_loaders):
            if len(client_val) == 0 or client_val is None:
                clients.append(Client(idx, Trainable(self.config, client_train, global_fed_model=False)))
            else:
                clients.append(Client(idx, Trainable(self.config, client_train, client_val, global_fed_model=False)))
        return clients

    def select_clients(self):
        """Randomly select a fraction of clients for training in this round"""
        num_clients_to_select = max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        self.selected_clients = random.sample(self.clients, num_clients_to_select)
        return self.selected_clients

    def train_selected_clients(self, logger, global_state_dict, loss_fn, communication_round, save_results, save_model):
        """Train all selected clients and return their updates"""
        client_deltas = []
        client_sizes = []

        for client in self.selected_clients:
            update, size = client.train_local_model(logger, global_state_dict, loss_fn, communication_round, save_results, save_model)
            client_deltas.append(update)
            client_sizes.append(size)

        return client_deltas, client_sizes


