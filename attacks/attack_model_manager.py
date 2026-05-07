import torch
from sklearn.metrics import accuracy_score

from attacks.shadow_model_manager import ShadowModelManager
from config import *
from models.trainable import Trainable
from training.train_single_model import train_single_model
from models.decision_tree import DecisionTreeWrapper


class AttackModelManager:
    def __init__(self, config: AttackConfig, save_model=False, save_results=False, plot_results=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        self.config = config
        self.attack_model_type = config.ATTACK_MODEL_TYPE

    def _create_attack_model(self, train_loader=None):
        """
        Create an attack model with architecture dynamically determined by config.LAYER_SIZES.
        The final layer always outputs 1 value for binary classification.
        """
        if self.attack_model_type == 'nn':
            if not hasattr(self.config, 'LAYER_SIZES') or len(self.config.LAYER_SIZES) < 2:
                raise ValueError("LAYER_SIZES must be defined in the config with at least two sizes (input and output).")
            elif train_loader is None:
                raise ValueError("train_loader must be defined for the NN attack model.")
            return Trainable(self.config, train_loader)
        elif self.attack_model_type == 'softmax':
            # Simple softmax classifier for binary classification
            input_size = self.config.TOP_K_FEATURES if hasattr(self.config, 'TOP_K_FEATURES') else 10
            self.config.LAYER_SIZES = [input_size, 1]
            if train_loader is None:
                raise ValueError("train_loader must be defined for the softmax attack model.")
            return Trainable(self.config, train_loader)
        elif self.attack_model_type == 'decision_tree':
            # Set the input size based on config or default
            input_size = self.config.TOP_K_FEATURES if hasattr(self.config, 'TOP_K_FEATURES') else 10
            return DecisionTreeWrapper(input_size)
        else:
            raise ValueError(f"Unsupported attack model type: {self.attack_model_type}. Supported types are 'nn', 'softmax' and 'decision_tree.")

    def train_shadow_models(self, logger, train_loader, val_loader):
        """
        Train shadow models to create the attack model dataset.

        :param logger: Logger for tracking training progress.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :return: Train and validation DataLoader for the attack model.
        """
        shadow_model_manager = ShadowModelManager(self.config.SHADOW_MODEL_CONFIG, train_loader, val_loader)
        shadow_model_manager.train(logger, save_results=self.save_results, save_model=self.save_model)
        attack_train_loader = shadow_model_manager.create_attack_dataset(k=self.config.TOP_K_FEATURES)
        return attack_train_loader

    def create_general_attack_model(self, logger, attack_train_loader):
        """
        Train a single attack model for membership inference across all classes.

        :param logger: Logger for tracking training progress.
        :param attack_train_loader: DataLoader containing attack training data (features, membership labels, class labels).
        :returns: A trained attack model for binary membership classification.
        """
        # Collect all data regardless of class
        all_features = []
        all_labels = []

        # Extract all features and membership labels
        for features, membership_labels, _ in attack_train_loader:
            batch_size = features.shape[0]

            for i in range(batch_size):
                all_features.append(features[i])
                all_labels.append(membership_labels[i].item())

        # Create tensors from collected data
        features = torch.stack(all_features)
        labels = torch.tensor(all_labels, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, labels),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

        # Train the attack model
        attack_model = self._create_attack_model(train_loader=train_loader)

        if isinstance(attack_model, DecisionTreeWrapper):
            # Convert back to integer labels for decision tree
            int_labels = labels.long()
            attack_model.fit(features, int_labels)
        else:
            # Use BCEWithLogitsLoss for neural networks (combines sigmoid + BCE)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            train_single_model(logger, attack_model, loss_fn, save_model=self.save_model, save_results=self.save_results)

        return attack_model


    def perform_attack(self, target_model, target_loader, attack_model):
        """
        Perform a membership inference attack on a target model using a data loader.

        :param target_model: Target model to be attacked.
        :type target_model: torch.nn.Module
        :param target_loader: DataLoader containing samples to test for membership
        :type target_loader: torch.utils.data.DataLoader
        :param attack_model: A single attack model
        :type attack_model: torch.nn.Module
        :returns: Predicted membership status (1=member, 0=non-member) and membership probabilities
        """
        target_model.eval()
        device = next(target_model.parameters()).device
        results = []
        probs = []

        # Process batches from the loader
        with torch.no_grad():
            for data, labels in target_loader:
                data = data.to(device)
                labels = labels.to(device)
                # Get predictions from the target model
                outputs = target_model(data)
                prob_vectors = torch.softmax(outputs, dim=1)

                batch_size = len(labels)

                # Extract top-k features for the entire batch
                top_k = min(self.config.TOP_K_FEATURES, prob_vectors.shape[1])
                topk_values, _ = torch.topk(prob_vectors, top_k, dim=1)

                # Get attack model predictions for the entire batch
                if isinstance(attack_model, DecisionTreeWrapper):
                    # Decision tree expects CPU tensors
                    attack_features_cpu = topk_values.cpu()
                    batch_predictions = []
                    batch_probs = []

                    for i in range(batch_size):
                        pred = attack_model.predict(attack_features_cpu[i].unsqueeze(0))
                        prob = attack_model.predict_proba(attack_features_cpu[i].unsqueeze(0))
                        batch_predictions.append(pred)
                        batch_probs.append(prob)

                    results.extend(batch_predictions)
                    probs.extend(batch_probs)
                else:
                    # Neural network attack model
                    attack_outputs = attack_model.model(topk_values)

                    # For BCEWithLogitsLoss training, apply sigmoid for inference
                    membership_probs = torch.sigmoid(attack_outputs.squeeze())
                    predictions = (membership_probs > 0.5).float()

                    results.extend(predictions.cpu().tolist())
                    probs.extend(membership_probs.cpu().tolist())

        return results, probs