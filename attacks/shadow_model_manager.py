from copy import copy

import torch
import torch.nn as nn

from models.logisitc_regression import LogisticRegression
from models.random_forest import RandomForest
from training.train_single_model import train_single_model
from training.evaluate_model import evaluate_model
from models.trainable import Trainable
from config import BaseConfig


class ShadowModelManager:
    def __init__(self, config: BaseConfig, member_loader, non_member_loader):
        self.config = config
        self.models = self._create_models(member_loader, non_member_loader)

    def _create_models(self, train_loader, val_loader):
        # Create a list of shadow models
        models = []
        num_models = getattr(self.config, 'NUM_SHADOW_MODELS', 1)
        for i in range(num_models):
            model = Trainable(self.config, train_loader, val_loader, global_fed_model=False, shadow_model=True)
            models.append(model)

        return models

    def train(self, logger, save_results, save_model):
        # Train the shadow model on the provided data
        if self.config.VERBOSE:
            print(f'Training Shadow Models on {self.config.DATASET} dataset...')
            eps = "Epsilon" if hasattr(self.config, 'EPSILON') else ""
            print("Shadow#", "Epoch", "Loss", "Accuracy (%)", "Precision", "Recall", "F1 Score", "Time (s)", "CPU (%)","", eps, sep="\t")
            print(100 * "-")
        for i, model in enumerate(self.models):
            loss_fn = nn.CrossEntropyLoss()
            train_single_model(logger, model, loss_fn, client_id=i, save_results=save_results, save_model=save_model)

            # evaluate_model(logger, model, loss_fn, model.val_loader, save_results=save_results)

    def create_attack_dataset(self, k=None):
        """
        Create a labeled dataset for training the attack model. For each shadow model collect the predictions on the
        training data and validation data, Label them as "in" (1) and "out" (0) respectively.

        :returns: Train dataset containing top_k probability vector with member label, and class labels.
        """
        # Calculate dataset sizes first to pre-allocate tensors
        member_size = sum(len(dataloader.dataset) for model in self.models for dataloader in [model.train_loader])
        non_member_size = sum(len(dataloader.dataset) for model in self.models for dataloader in [model.val_loader])
        total_size = member_size + non_member_size

        # Determine feature dimension based on k
        sample_batch = next(iter(self.models[0].train_loader))[0]
        output_dim = self.models[0].model(sample_batch[:1]).shape[1]
        feature_dim = k if k is not None and k <= output_dim else output_dim

        # Pre-allocate tensors
        attack_features = torch.zeros(total_size, feature_dim)
        attack_labels = torch.zeros(total_size, dtype=torch.long)
        class_labels = torch.zeros(total_size, dtype=torch.long)

        idx = 0
        for model in self.models:
            model.model.eval()  # Set to evaluation mode

            # Process training data (label=1) and validation data (label=0)
            for is_train, dataloader in [(1, model.train_loader), (0, model.val_loader)]:
                with torch.no_grad():
                    for inputs, targets in dataloader:
                        inputs = inputs.to(model.device)
                        targets = targets.to(model.device)
                        batch_size = inputs.size(0)
                        outputs = model.model.forward(inputs)
                        probs = torch.softmax(outputs, dim=1)

                        if k is not None and k <= probs.size(1):
                            features, _ = torch.topk(probs, k, dim=1)
                        else:
                            features = probs

                        # Store in pre-allocated tensors
                        attack_features[idx:idx + batch_size] = features
                        attack_labels[idx:idx + batch_size] = is_train
                        class_labels[idx:idx + batch_size] = targets

                        idx += batch_size

        train_dataset = torch.utils.data.TensorDataset(attack_features, attack_labels, class_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        return train_loader