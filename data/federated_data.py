import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split


class FederatedDataManager:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.sampling_strategy = config.SAMPLING_STRATEGY if hasattr(config, 'SAMPLING_STRATEGY') else 'balanced'

    def load_and_split_data(self, val_split=0.1, client_sizes=None, plot_distribution=False):
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

        # Verify client_sizes if provided
        if client_sizes is not None:
            assert len(client_sizes) == self.config.NUM_CLIENTS, "Must provide size for each client"

        # Create client data splits
        if self.config.IID:
            client_datasets = self._create_iid_splits(train_dataset, client_sizes)
        else:
            client_datasets = self._create_non_iid_splits(train_dataset, client_sizes)

        if plot_distribution: self._plot_client_data_distribution(client_datasets)

        # Create loaders
        client_loaders = self._get_client_dataloaders(client_datasets, val_split)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        return client_loaders, test_loader

    def _create_iid_splits(self, dataset, client_sizes=None):
        """Split dataset IID among clients

        Args:
            dataset: The full dataset to split
            client_sizes: Optional list specifying a desired dataset size for each client
        """
        indices = torch.randperm(len(dataset)).tolist()
        client_datasets = []

        if client_sizes is None:
            # Default behavior: split evenly among clients
            num_items_per_client = len(dataset) // self.config.NUM_CLIENTS

            for i in range(self.config.NUM_CLIENTS):
                start_idx = i * num_items_per_client
                end_idx = (i + 1) * num_items_per_client if i < self.config.NUM_CLIENTS - 1 else len(dataset)
                client_indices = indices[start_idx:end_idx]
                client_datasets.append(Subset(dataset, client_indices))
        else:
            # Custom sizes with different sampling strategies
            for i, size in enumerate(client_sizes):
                client_indices = self._sample_indices(indices, size)
                client_datasets.append(Subset(dataset, client_indices))

        return client_datasets

    def _create_non_iid_splits(self, dataset, client_sizes=None):
        """Split dataset non-IID among clients (each client gets mostly 2 classes)

        Args:
            dataset: The full dataset to split
            client_sizes: Optional list specifying a desired dataset size for each client
        """
        labels = dataset.targets.clone().detach()
        client_datasets = []

        # Sort data by labels
        sorted_indices = torch.argsort(labels).tolist()

        # Create a list to track allocated indices
        allocated = [False] * len(dataset)

        # Each client gets primarily two random classes but also some examples from other classes
        classes_per_client = 2
        client_class_indices = {}

        # First, collect potential indices for each client based on classes
        for client_idx in range(self.config.NUM_CLIENTS):
            # Assign primary classes to this client (with overlap between clients)
            primary_classes = torch.randperm(10)[:classes_per_client].tolist()
            client_class_indices[client_idx] = []

            # Get indices for primary classes
            for cls in primary_classes:
                cls_indices = [idx for idx in sorted_indices if labels[idx] == cls]

                # Take 80% of available samples from this class
                num_samples = int(0.8 * len(cls_indices) / (self.config.NUM_CLIENTS / classes_per_client))
                available_indices = [idx for idx in cls_indices if not allocated[idx]]

                if len(available_indices) > num_samples:
                    selected_indices = available_indices[:num_samples]
                    client_class_indices[client_idx].extend(selected_indices)

                    # Mark as allocated
                    for idx in selected_indices:
                        allocated[idx] = True

            # Add some random samples from other classes for diversity
            other_indices = [i for i in range(len(dataset)) if not allocated[i]]
            if other_indices:
                num_other = int(len(client_class_indices[client_idx]) * 0.2)  # 20% from other classes
                if num_other > len(other_indices):
                    num_other = len(other_indices)

                other_selected = np.random.choice(other_indices, num_other, replace=False).tolist()
                client_class_indices[client_idx].extend(other_selected)

                # Mark as allocated
                for idx in other_selected:
                    allocated[idx] = True

        # Now adjust sizes if client_sizes is provided
        if client_sizes is not None:
            resized_indices = []

            for client_idx, target_size in enumerate(client_sizes):
                base_indices = client_class_indices[client_idx]
                client_indices = self._sample_indices_with_class_distribution(base_indices, target_size, labels)
                resized_indices.append(client_indices)

            # Create client datasets with resized indices
            for indices in resized_indices:
                client_datasets.append(Subset(dataset, indices))
        else:
            # Use original indices
            for client_idx in range(self.config.NUM_CLIENTS):
                client_datasets.append(Subset(dataset, client_class_indices[client_idx]))

        return client_datasets

    def _sample_indices(self, indices, target_size):
        """Sample indices to create a dataset of target size

        Args:
            indices: Pool of indices to sample from
            target_size: Desired size of the resulting dataset

        Returns:
            List of sampled indices
        """
        if target_size <= len(indices):
            # If a target size is smaller than available indices, sample without replacement
            return np.random.choice(indices, target_size, replace=False).tolist()
        else:
            # If a target size is larger, use the sampling strategy specified
            if self.sampling_strategy == 'replace':
                # Pure sampling with replacement
                return np.random.choice(indices, target_size, replace=True).tolist()
            elif self.sampling_strategy == 'augment':
                # Use all original data and augmented samples (recommended approach)
                # This preserves all original data while adding augmented samples
                additional_needed = target_size - len(indices)
                augment_indices = np.random.choice(indices, additional_needed, replace=True).tolist()
                return indices + augment_indices
            elif self.sampling_strategy == 'balanced':
                # Balance original data with augmented data in the ratio defined in config
                # or default to 1:1 if not specified
                balance_ratio = getattr(self.config, 'BALANCE_RATIO', 1.0)
                original_count = min(len(indices), int(target_size / (1 + balance_ratio)))
                sampled_original = np.random.choice(indices, original_count, replace=False).tolist()

                augment_count = target_size - original_count
                augment_indices = np.random.choice(indices, augment_count, replace=True).tolist()

                return sampled_original + augment_indices
            else:
                # Default to augment strategy
                additional_needed = target_size - len(indices)
                augment_indices = np.random.choice(indices, additional_needed, replace=True).tolist()
                return indices + augment_indices

    def _sample_indices_with_class_distribution(self, indices, target_size, labels):
        """Sample indices while preserving class distribution

        Args:
            indices: Base indices to sample from
            target_size: Desired size of the resulting dataset
            labels: Tensor of labels for the dataset

        Returns:
            List of sampled indices that maintain class distribution
        """
        if target_size <= len(indices):
            # If the target is smaller, just sample without replacement
            return np.random.choice(indices, target_size, replace=False).tolist()

        # Get class distribution from original indices
        class_counts = {}
        for idx in indices:
            label = labels[idx].item()
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(indices)
        class_ratios = {label: count / total for label, count in class_counts.items()}

        # Calculate target counts for each class
        target_counts = {label: max(1, int(ratio * target_size)) for label, ratio in class_ratios.items()}

        # Adjust to ensure we get exactly target_size samples
        total_allocated = sum(target_counts.values())
        if total_allocated < target_size:
            # Distribute the remaining samples proportionally
            remaining = target_size - total_allocated
            for label in sorted(class_ratios, key=class_ratios.get, reverse=True):
                if remaining <= 0:
                    break
                target_counts[label] += 1
                remaining -= 1

        # Create indices by class
        class_indices = {}
        for idx in indices:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Sample from each class
        result_indices = []
        for label, target_count in target_counts.items():
            source_indices = class_indices.get(label, [])
            if not source_indices:
                continue

            if len(source_indices) >= target_count:
                # Enough samples, just take without replacement
                sampled = np.random.choice(source_indices, target_count, replace=False).tolist()
            else:
                # Not enough samples, use replacement strategy
                if self.sampling_strategy == 'replace':
                    sampled = np.random.choice(source_indices, target_count, replace=True).tolist()
                else:
                    # Start with all original samples
                    sampled = source_indices.copy()
                    # Add augmented samples
                    remaining = target_count - len(sampled)
                    augmented = np.random.choice(source_indices, remaining, replace=True).tolist()
                    sampled.extend(augmented)

            result_indices.extend(sampled)

        # Shuffle final result
        np.random.shuffle(result_indices)
        return result_indices

    def _get_client_dataloaders(self, client_datasets, val_split=0.1):
        """Create training and validation DataLoaders for each client"""
        client_dataloaders = []

        for dataset in client_datasets:
            # Calculate the split sizes
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            # Split the dataset into training and validation
            train_subset, val_subset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )

            # Create loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False  # No need to shuffle validation data
            )

            # Append tuple of (training_loader, val_loader)
            client_dataloaders.append((train_loader, val_loader))

        return client_dataloaders


    @staticmethod
    def _plot_client_data_distribution(client_datasets):
        """
        Plot the distribution of labels across clients in a 3D bar plot.

        Args:
            client_datasets: List of datasets, one for each client.
        """
        # Collect label distributions for each client
        num_clients = len(client_datasets)
        label_counts = []
        for client_dataset in client_datasets:
            labels = [label for _, label in client_dataset]
            counts = np.bincount(labels, minlength=10)  # Assuming 10 classes (0-9)
            label_counts.append(counts)

        label_counts = np.array(label_counts)

        # Create a 3D bar plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Define x, y, and z positions
        x_labels = np.arange(10)  # Labels (0-9)
        y_clients = np.arange(num_clients)  # Clients
        x_pos, y_pos = np.meshgrid(x_labels, y_clients, indexing="ij")
        x_pos = x_pos.ravel()
        y_pos = y_pos.ravel()
        z_pos = np.zeros_like(x_pos)

        # Heights of the bars (frequencies)
        heights = label_counts.T.ravel()

        # Width and depth of the bars
        dx = dy = 0.8

        # Plot the bars
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, heights, shade=True)

        # Set axis labels
        ax.set_xlabel("Label")
        ax.set_ylabel("Client")
        ax.set_zlabel("Frequency")
        ax.set_title("Label Distribution Across Clients")

        # Show the plot
        plt.show()