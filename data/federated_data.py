import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset


class FederatedDataManager:
    def __init__(self, config, dataset='mnist', transform=False):
        self.config = config
        self.generator = torch.Generator().manual_seed(config.SEED)
        # Separate transforms for training data (with optional augmentation)
        # and test data (standardToTensor + normalization)
        if transform:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Standard transform for training data without augmentation
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ])
        if dataset == 'emnist':
            self.full_train = datasets.EMNIST(root='./data', train=True, download=True, transform=transform, split='byclass')
            self.test_ds = datasets.EMNIST(root='./data', train=False, download=True, split='byclass', transform=transforms.ToTensor())
        elif dataset == 'mnist':
            self.full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            self.test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'fashion':
            self.full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            self.test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError("Unsupported dataset. Current datasets include 'mnist', 'emnist' or 'fashion'.")

        self.sampling_strategy = config.SAMPLING_STRATEGY if hasattr(config, 'SAMPLING_STRATEGY') else 'augment'

    def get_client_loaders(self, val_split=0.1, client_sizes=None, plot_distribution=False):

        # Verify client_sizes if provided
        if client_sizes is not None:
            assert len(client_sizes) == self.config.NUM_CLIENTS, "You must provide a size for each client"

        # Create client data splits
        if self.config.IID:
            client_datasets = self._create_iid_splits(self.full_train, client_sizes)
        else:
            client_datasets = self._create_non_iid_splits(self.full_train, client_sizes)

        if plot_distribution: self._plot_client_data_distribution(client_datasets)

        # Create loaders
        client_loaders = self._get_client_dataloaders(client_datasets, val_split)
        test_loader = DataLoader(self.test_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)

        return client_loaders, test_loader

    def get_privacy_experiment_shadow_loaders(self, client_sizes=None, plot_distribution=False):
        # Verify client_sizes if provided
        if client_sizes is not None:
            assert len(client_sizes) == self.config.NUM_CLIENTS, "You must provide a size for each client"

        full_dataset = ConcatDataset([self.full_train, self.test_ds])
        target_full_size = int(len(full_dataset) * 0.6)
        shadow_full_size = len(full_dataset) - target_full_size
        target_full_data, shadow_full_data = random_split(full_dataset, [target_full_size, shadow_full_size],generator=self.generator)
        # Split into member and non-member datasets
        target_member_size = int(len(target_full_data) * 0.5)
        target_non_member_size = len(target_full_data) - target_member_size
        shadow_member_size = int(len(shadow_full_data) * 0.5)
        shadow_non_member_size = len(shadow_full_data) - shadow_member_size
        target_member_data, target_non_member_data = random_split(target_full_data,[target_member_size, target_non_member_size], generator=self.generator)
        shadow_member_data, shadow_non_member_data = random_split(shadow_full_data,[shadow_member_size, shadow_non_member_size],generator=self.generator)

        # Create client data splits
        if self.config.IID:
            client_datasets = self._create_iid_splits(target_member_data, client_sizes)
        else:
            client_datasets = self._create_non_iid_splits(target_member_data, client_sizes)

        if plot_distribution: self._plot_client_data_distribution(client_datasets)

        full_target_train_loaders = DataLoader(target_member_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        client_loaders = self._get_client_dataloaders(client_datasets, 0.0)
        target_test_loader = DataLoader(target_non_member_data, batch_size=self.config.BATCH_SIZE, shuffle=False)
        shadow_member_loader = DataLoader(shadow_member_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        shadow_non_member_loader = DataLoader(shadow_non_member_data, batch_size=self.config.BATCH_SIZE, shuffle=False)

        return full_target_train_loaders, client_loaders, target_test_loader, (shadow_member_loader, shadow_non_member_loader)


    def _create_iid_splits(self, dataset, client_sizes):
        """Split dataset IID among clients


        :param dataset: The dataset to split
        :type dataset: torchvision.data.Dataset
        :param client_sizes: Optional list specifying a desired dataset size for each client
        """
        indices = torch.randperm(len(dataset), generator=self.generator).tolist()
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
                # Sample from the global pool of indices
                client_indices = self._sample_indices(indices, size)
                client_datasets.append(Subset(dataset, client_indices))

        return client_datasets

    def _create_non_iid_splits(self, dataset, client_sizes):
        """Split dataset non-IID among clients (each client gets mostly 2 classes)


        :param dataset: The dataset to split
        :type dataset: torchvision.data.Dataset
        :param client_sizes: Optional list specifying a desired dataset size for each client
        """
        # Ensure labels are on CPU for numpy operations
        labels = dataset.targets.clone().detach().cpu()
        client_datasets = []

        # Sort data by labels
        sorted_indices = torch.argsort(labels).tolist()

        # Create a list to track allocated indices from the sorted list
        allocated_sorted_indices = [False] * len(sorted_indices)

        # Each client gets primarily two random classes but also some examples from other classes
        # Make classes_per_client configurable? For now, keep at 2.
        classes_per_client = max(1, min(10, int(10 / (self.config.NUM_CLIENTS ** 0.5))))  # Scale with inverse square root of client count
        client_initial_indices = {} # This will store the base non-IID indices before resizing

        # First, collect initial potential indices for each client based on classes
        for client_idx in range(self.config.NUM_CLIENTS):
            # Assign primary classes to this client (with overlap between clients)
            primary_classes = torch.multinomial(torch.ones(10), classes_per_client, replacement=False,generator=self.generator).tolist()

            client_initial_indices[client_idx] = []

            # Get indices for primary classes
            for cls in primary_classes:
                # Find indices for this class within the sorted list
                # We need to find the range in sorted_indices corresponding to this class
                # A more robust way is to filter directly
                cls_sorted_indices = [idx for idx in sorted_indices if labels[idx] == cls]

                # Find available indices *for this class* that haven't been primarily allocated yet
                # by checking allocation status in sorted_indices list
                available_cls_indices = [original_idx for original_idx in cls_sorted_indices if not allocated_sorted_indices[sorted_indices.index(original_idx)]]

                # Take a portion of available samples from this class for primary distribution
                # The denominator (self.config.NUM_CLIENTS / classes_per_client) estimates how many clients share a primary class
                estimated_sharing_clients = self.config.NUM_CLIENTS / classes_per_client # This is an average, actual overlap varies
                if estimated_sharing_clients < 1: estimated_sharing_clients = 1 # Handle cases where classes_per_client > NUM_CLIENTS

                num_samples_from_class = int(0.8 * len(cls_sorted_indices) / estimated_sharing_clients)
                num_samples_from_class = max(1, num_samples_from_class) # Ensure at least 1 sample if possible

                # Ensure we don't request more than available
                num_samples_to_take = min(num_samples_from_class, len(available_cls_indices))

                if num_samples_to_take > 0:
                    # Sample randomly from available indices instead of slicing
                    selected_indices_from_class = np.random.choice(available_cls_indices, num_samples_to_take, replace=False).tolist()
                    client_initial_indices[client_idx].extend(selected_indices_from_class)

                    # Mark these selected indices as allocated in the global sorted list tracker
                    for original_idx in selected_indices_from_class:
                         allocated_sorted_indices[sorted_indices.index(original_idx)] = True

        # Add some random samples from other classes for diversity
        # Collect indices that haven't been allocated as primary samples
        unallocated_indices = [original_idx for i, original_idx in enumerate(sorted_indices) if not allocated_sorted_indices[i]]

        for client_idx in range(self.config.NUM_CLIENTS):
            current_client_size = len(client_initial_indices[client_idx])
            if current_client_size == 0: # Handle cases where a client got no primary samples
                 num_other = int(len(dataset) * 0.01) # Give at least some minimum samples
            else:
                 num_other = int(current_client_size * 0.2) # Add 20% from other classes

            # Ensure we don't take more "other" samples than available globally
            num_other_to_take = min(num_other, len(unallocated_indices))

            if num_other_to_take > 0:
                 # Sample without replacement from the remaining unallocated pool
                 # Note: These 'other' samples *are* also marked as allocated so they aren't reused
                 # This makes the overall split disjoint before potential resampling
                 selected_other_indices = np.random.choice(unallocated_indices, num_other_to_take, replace=False).tolist()
                 client_initial_indices[client_idx].extend(selected_other_indices)

                 # Remove selected indices from the pool of unallocated indices for subsequent clients
                 unallocated_indices = [idx for idx in unallocated_indices if idx not in selected_other_indices]

        # Now adjust sizes if client_sizes is provided by resampling/augmenting from the initial distribution
        final_client_indices = []
        for client_idx in range(self.config.NUM_CLIENTS):
             initial_indices = client_initial_indices[client_idx]
             if client_sizes is not None:
                 target_size = client_sizes[client_idx]
                 # Resample/augment from the initial set of indices while preserving *its* class distribution
                 client_indices = self._sample_indices_with_class_distribution(initial_indices, target_size, labels)
             else:
                 # Use the initial indices determined by the non-IID logic
                 client_indices = initial_indices

             final_client_indices.append(client_indices)

        # Create client datasets from the final set of indices
        for indices in final_client_indices:
            client_datasets.append(Subset(dataset, indices))

        return client_datasets


    def _sample_indices(self, indices, target_size):
        """Sample indices to create a dataset of target size.
           Used primarily when target_size > len(indices).

        :param indices: Pool of indices to sample from
        :type indices: list or np.ndarray
        :param target_size: Desired size of the resulting dataset
        :type target_size: int
        :returns: List of sampled indices
        """
        # Ensure indices is a list or numpy array for random.choice
        indices = list(indices) if not isinstance(indices, (list, np.ndarray)) else indices

        if target_size <= len(indices):
            # If a target size is smaller than available indices, sample without replacement
            return np.random.choice(indices, target_size, replace=False).tolist()
        else:
            # If a target size is larger, use the sampling strategy specified
            additional_needed = target_size - len(indices)

            if self.sampling_strategy == 'replace':
                # Pure sampling with replacement from the original pool
                return np.random.choice(indices, target_size, replace=True).tolist()
            elif self.sampling_strategy == 'augment':
                # Use all original data and add augmented samples (repetitions)
                # This preserves all original data while adding augmented samples
                if len(indices) == 0: # Handle empty input indices
                     return []
                augment_indices = np.random.choice(indices, additional_needed, replace=True).tolist()
                return indices + augment_indices
            elif self.sampling_strategy == 'balanced':
                # Balance original data with augmented data based on ratio
                # This strategy samples a *subset* of original data + augmented data
                balance_ratio = getattr(self.config, 'BALANCE_RATIO', 1.0)
                # Calculate how many unique original samples to include
                # target_size = original_count + augment_count
                # augment_count = balance_ratio * original_count
                # target_size = original_count * (1 + balance_ratio)
                original_count = min(len(indices), int(target_size / (1 + balance_ratio)))
                original_count = max(0, original_count) # Ensure not negative

                # Sample 'original_count' samples without replacement
                sampled_original = np.random.choice(indices, original_count, replace=False).tolist()

                # Sample the rest 'with' replacement from the full pool
                augment_count = target_size - original_count
                if len(indices) == 0 and augment_count > 0: return [] # Cannot augment from empty pool
                augment_indices = np.random.choice(indices, augment_count, replace=True).tolist()

                return sampled_original + augment_indices
            else:
                # Default to augment strategy if strategy is unknown
                if len(indices) == 0 and additional_needed > 0: return []
                augment_indices = np.random.choice(indices, additional_needed, replace=True).tolist()
                return indices + augment_indices

    def _sample_indices_with_class_distribution(self, indices, target_size, labels):
        """Sample indices while preserving class distribution of the input indices.
           Used when target_size > len(indices).

        Args:
            indices: Base indices to sample from (list or np.ndarray)
            target_size: Desired size of the resulting dataset
            labels: Tensor of labels for the full dataset

        Returns:
            List of sampled indices that maintain the class distribution of the input indices.
        """
        # Ensure indices is a list or numpy array
        indices = list(indices) if not isinstance(indices, (list, np.ndarray)) else indices

        if target_size <= len(indices):
            # If the target is smaller, just sample without replacement
            return np.random.choice(indices, target_size, replace=False).tolist()

        if len(indices) == 0:
            return [] # Cannot sample from empty list

        # Get class distribution from original indices
        class_counts = {}
        for idx in indices:
            # Ensure index is within bounds of labels tensor
            if idx < len(labels):
                label = labels[idx].item()
                class_counts[label] = class_counts.get(label, 0) + 1

        # Handle case where no valid indices were found
        if not class_counts:
            return []

        total = len(indices)
        class_ratios = {label: count / total for label, count in class_counts.items()}

        # Calculate target counts for each class based on the desired size
        target_counts = {label: int(ratio * target_size) for label, ratio in class_ratios.items()}

        # Ensure every class represented in original indices gets at least 1 sample in target_counts, if target_size allows
        # Also ensure sum is exactly target_size
        total_allocated = sum(target_counts.values())

        # First, ensure minimum 1 for represented classes, then adjust total
        for label in class_counts.keys():
            if target_counts[label] == 0 and target_size > 0:
                 target_counts[label] = 1
        total_allocated = sum(target_counts.values()) # Recalculate after min(1) adjustment

        # Distribute remaining samples if total_allocated is less than target_size
        remaining = target_size - total_allocated
        if remaining > 0:
             # Distribute based on highest ratios (or just add to any class)
             # Adding to classes with highest ratios is a common approach
             sorted_labels = sorted(class_ratios, key=class_ratios.get, reverse=True)
             label_idx = 0
             while remaining > 0 and label_idx < len(sorted_labels):
                  label = sorted_labels[label_idx]
                  target_counts[label] += 1
                  remaining -= 1
                  label_idx += 1
                  if label_idx == len(sorted_labels): # Loop back if needed
                      label_idx = 0

        # Ensure total_allocated doesn't exceed target_size (can happen with min(1) and small target_size)
        # If it exceeds, trim from classes with the lowest counts (or just remove samples later)
        # A simpler approach if it exceeds is to just sample exactly target_size at the end
        # However, aiming for precise counts first is better. Let's cap if needed.
        if sum(target_counts.values()) > target_size:
             excess = sum(target_counts.values()) - target_size
             # Remove from classes with lowest counts (that are > 1)
             sorted_labels_asc = sorted(class_counts, key=class_counts.get, reverse=False)
             label_idx = 0
             while excess > 0 and label_idx < len(sorted_labels_asc):
                  label = sorted_labels_asc[label_idx]
                  if target_counts[label] > 1:
                      remove_count = min(excess, target_counts[label] - 1)
                      target_counts[label] -= remove_count
                      excess -= remove_count
                  label_idx += 1


        # Create indices by class from the input 'indices' pool
        class_indices_pool = {}
        for idx in indices:
             if idx < len(labels):
                 label = labels[idx].item()
                 if label not in class_indices_pool:
                     class_indices_pool[label] = []
                 class_indices_pool[label].append(idx)

        # Sample from each class to meet the target_counts
        result_indices = []
        for label, target_count in target_counts.items():
            source_indices = class_indices_pool.get(label, [])
            if not source_indices or target_count == 0:
                continue

            if len(source_indices) >= target_count:
                # Enough samples, just take without replacement
                sampled = np.random.choice(source_indices, target_count, replace=False).tolist()
            else:
                # Not enough samples, use replacement strategy to reach target_count
                if self.sampling_strategy == 'replace':
                    sampled = np.random.choice(source_indices, target_count, replace=True).tolist()
                else: # 'augment', 'balanced', or any other strategy defaults to augment-like
                    # Start with all original samples for this class
                    sampled = source_indices.copy()
                    # Add augmented samples (repetitions) to reach the target count
                    remaining_needed = target_count - len(sampled)
                    if remaining_needed > 0: # Only augment if more needed
                       augmented = np.random.choice(source_indices, remaining_needed, replace=True).tolist()
                       sampled.extend(augmented)

            result_indices.extend(sampled)

        # Ensure the final list has exactly target_size elements
        # This can sometimes be slightly off due to rounding in target_counts or issues in logic above
        # A robust way is to trim or pad the result
        if len(result_indices) > target_size:
            result_indices = result_indices[:target_size] # Trim excess (simple approach, might slightly affect final ratio)
            np.random.shuffle(result_indices) # Shuffle after trimming
        elif len(result_indices) < target_size:
             # Pad by sampling with replacement from the current result_indices
             if len(result_indices) == 0: return [] # Cannot pad empty list
             needed = target_size - len(result_indices)
             padding = np.random.choice(result_indices, needed, replace=True).tolist()
             result_indices.extend(padding)
             np.random.shuffle(result_indices) # Shuffle after padding
        else:
            # If exactly target_size, just shuffle
            np.random.shuffle(result_indices)


        return result_indices

    def _get_client_dataloaders(self, client_datasets, val_split):
        """Create training and validation DataLoaders for each client"""
        client_dataloaders = []

        for dataset in client_datasets:
            # Calculate the split sizes
            # Ensure val_size is at least 1 if the dataset size is > 0 and val_split > 0
            # And ensure train_size is > 0 if the dataset size is > 0
            total_size = len(dataset)
            if total_size == 0:
                 train_size = 0
                 val_size = 0
            else:
                 val_size = int(total_size * val_split)
                 # Ensure val_size is at least 1 if total_size > 0 and val_split > 0, unless total_size is 1
                 if val_size == 0 and val_split > 0 and total_size > 1:
                     val_size = 1
                 train_size = total_size - val_size
                 # Ensure train_size is at least 1 if total_size > 0 and val_size < total_size
                 if train_size == 0 and total_size > 0 and val_size < total_size:
                      train_size = 1
                      val_size = total_size - train_size # Adjust val_size accordingly

            # Handle edge case where total_size is 1
            if total_size == 1 and val_split > 0:
                train_size = 0
                val_size = 1
            elif total_size == 1 and val_split == 0:
                 train_size = 1
                 val_size = 0

            if train_size + val_size != total_size:
                 # Fallback or error handling if split sizes calculation is weird
                 # Simple fallback: put all into train if split fails
                 train_size = total_size
                 val_size = 0
                 # print(f"Warning: Split sizes {train_size}+{val_size} != total size {total_size} for a client. Using train={total_size}, val=0.")

            # Split the dataset into training and validation only if sizes are valid
            if train_size > 0 or val_size > 0:
                # Ensure a generator is used for reproducibility
                # Check if sizes are valid for random_split
                if train_size + val_size != len(dataset):
                     print(f"Warning: Train+Val sizes ({train_size}+{val_size}) don't match dataset size ({len(dataset)}). Skipping split for this client.")
                     train_subset = dataset # Put all data into train if split is problematic
                     val_subset = Subset(dataset, []) # Empty val set
                else:
                    try:
                         train_subset, val_subset = random_split(dataset,[train_size, val_size],generator=self.generator)
                    except ValueError as e:
                         print(f"Error during random_split with sizes [{train_size}, {val_size}] for dataset size {len(dataset)}: {e}")
                         print("Putting all data into train subset.")
                         train_subset = dataset # Put all data into train if split fails
                         val_subset = Subset(dataset, []) # Empty val set

            else:
                 # Dataset is empty
                 train_subset = Subset(dataset, [])
                 val_subset = Subset(dataset, [])

            # Create loaders
            # Use shuffle=False if train_subset is empty to avoid DataLoader errors
            train_loader = DataLoader(train_subset, batch_size=self.config.BATCH_SIZE, shuffle=True and len(train_subset) > 0)

            val_loader = DataLoader(val_subset, batch_size=self.config.BATCH_SIZE, shuffle=False)  # No need to shuffle validation data

            # Append tuple of (train_loader, val_loader)
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
            # Access the underlying dataset's targets if it's a Subset
            if isinstance(client_dataset, Subset):
                 # Need to map subset indices to original dataset labels
                 # This requires accessing the original dataset, which Subset doesn't directly expose easily
                 # A simpler way assuming MNIST-like structure or adding labels attribute
                 # Let's iterate through subset indices and get labels from the base dataset
                 base_dataset = client_dataset.dataset
                 indices = client_dataset.indices
                 if hasattr(base_dataset, 'targets'):
                     labels = base_dataset.targets[indices].cpu().tolist()
                 else:
                     # Fallback: iterate through actual data points (can be slow)
                     print("Warning: Could not directly access targets from subset. Iterating data points for plotting.")
                     try:
                         labels = [label for _, label in client_dataset]
                     except Exception as e:
                         print(f"Error iterating dataset for plotting: {e}. Skipping plot for this client.")
                         labels = [] # Skip plotting for this client

            else: # Not a Subset, assume it's the full dataset or similar structure
                if hasattr(client_dataset, 'targets'):
                    labels = client_dataset.targets.cpu().tolist()
                else:
                    # Fallback: iterate through actual data points (can be slow)
                    print("Warning: Could not directly access targets. Iterating data points for plotting.")
                    try:
                        labels = [label for _, label in client_dataset]
                    except Exception as e:
                         print(f"Error iterating dataset for plotting: {e}. Skipping plot for this client.")
                         labels = [] # Skip plotting for this client


            counts = np.bincount(labels, minlength=10)  # Assuming 10 classes (0-9) for MNIST
            label_counts.append(counts)

        label_counts = np.array(label_counts)

        # Create a 3D bar plot only if there is data to plot
        if num_clients > 0 and label_counts.shape[1] > 0 and np.sum(label_counts) > 0:
             fig = plt.figure(figsize=(10, 7))
             ax = fig.add_subplot(111, projection='3d')

             # Define x, y, and z positions
             num_labels = label_counts.shape[1]
             x_labels_pos = np.arange(num_labels)  # Labels (0-9)
             y_clients_pos = np.arange(num_clients)  # Clients
             x_pos, y_pos = np.meshgrid(x_labels_pos, y_clients_pos, indexing="ij")
             x_pos = x_pos.ravel()
             y_pos = y_pos.ravel()
             z_pos = np.zeros_like(x_pos)

             # Heights of the bars (frequencies)
             heights = label_counts.T.ravel()

             # Remove bars with zero height for cleaner plot
             non_zero_mask = heights > 0
             x_pos = x_pos[non_zero_mask]
             y_pos = y_pos[non_zero_mask]
             z_pos = z_pos[non_zero_mask]
             heights = heights[non_zero_mask]


             # Width and depth of the bars
             dx = dy = 0.8

             # Plot the bars
             ax.bar3d(x_pos, y_pos, z_pos, dx, dy, heights, shade=True)

             # Set axis labels and ticks
             ax.set_xlabel("Label")
             ax.set_ylabel("Client")
             ax.set_zlabel("Frequency")
             ax.set_title("Label Distribution Across Clients")

             ax.set_xticks(x_labels_pos + dx/2) # Center ticks on bars
             ax.set_xticklabels(x_labels_pos)
             ax.set_yticks(y_clients_pos + dy/2) # Center ticks on bars
             ax.set_yticklabels(y_clients_pos)


             plt.show()
        else:
            print("No data or clients to plot distribution.")