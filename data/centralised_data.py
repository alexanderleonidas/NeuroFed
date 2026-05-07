import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import pandas as pd

class CentralisedDataManager:
    """
    Centralised Data Manager for MNIST and EMNIST datasets.

    :param batch_size: Batch size for DataLoader.
    :type batch_size: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :param dataset: String indicating which dataset to use ('mnist' or 'emnist'). Defaults to 'mnist'.
    :type dataset: string, optional
    :param transform: Type of transformation to apply to the dataset ('augmented', 'noised', or None).
    :type transform: string, optional
    """
    def __init__(self, batch_size: int, seed: int, dataset='mnist', transform=None):
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        if transform == 'augmented':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif transform == 'noised':
            noise_level = 0.2
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.clamp(x + noise_level * torch.randn_like(x), 0.0, 1.0)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset_name = dataset
        self.full_train, self.test_ds = self._get_dataset(dataset)

    def _get_dataset(self, dataset):
        if dataset == 'emnist':
            full_train = datasets.EMNIST(root='./data', train=True, download=True, transform=self.transform, split='byclass')
            test_ds = datasets.EMNIST(root='./data', train=False, download=True, split='byclass', transform=transforms.ToTensor())
        elif dataset == 'mnist':
            full_train = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'fashion':
            full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=self.transform)
            test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'purchase':
            raise NotImplementedError("Purchase dataset loading is not implemented. Use another dataset please")
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist', 'emnist' or 'fashion'.")

        return full_train, test_ds

    def get_loaders(self, val_split=None, train_size=None):
        """
        Prepare DataLoaders for training and testing datasets. If val_split is specified, it splits the training data.

        :return: Tuple of (train_loader, val_loader, test_loader) if val_split is specified, otherwise (train_loader, None, test_loader).
        """
        if train_size is not None:
            if isinstance(train_size, float) and 0 < train_size <= 1.0:
                train_size = int(len(self.full_train) * train_size)
            else:
                raise ValueError("train_size must be a float between 0 and 1 or a positive integer.")
            self.full_train, _ = random_split(self.full_train, [train_size, len(self.full_train) - train_size], generator=self.generator)

        if val_split is not None and val_split > 0:
            val_size = int(len(self.full_train) * val_split)
            train_size = len(self.full_train) - val_size
            train_ds, val_ds = random_split(self.full_train, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            train_loader = DataLoader(self.full_train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
            return train_loader, None, test_loader

    def get_privacy_experiment_shadow_loaders(self, data_size=None, transfer_attack=False):
        """
        Prepare loaders for privacy experiments, splitting the full chosen dataset into 4 equal partitions for the
        target and shadow model.

        :param data_size: If specified, it can be a float between 0 and 1 to indicate the proportion of the dataset to use
        :type data_size: float, optional
        :param transfer_attack: If True, uses a different dataset for the shadow model to simulate a transfer attack.
        :type transfer_attack: bool, optional
        :return: Tuple of (target_model_loaders, shadow_model_loaders) or (None, shadow_model_loaders) if with_target is False.
        """

        target_full_data = ConcatDataset([self.full_train, self.test_ds])
        shadow_dataset_name = None
        if transfer_attack:
            available_datasets = ['mnist', 'fashion', 'emnist']
            # Exclude the current dataset to pick a different one for the shadow model
            shadow_dataset_name = next((ds for ds in available_datasets if ds != self.dataset_name), None)
            if shadow_dataset_name is None:
                raise ValueError("Could not find a different dataset for transfer attack.")

            shadow_train, shadow_test = self._get_dataset(shadow_dataset_name)
            shadow_full_data = ConcatDataset([shadow_train, shadow_test])
        else:
            # Split into target and shadow sets
            target_full_size = int(len(target_full_data) * 0.5)
            shadow_full_size = len(target_full_data) - target_full_size
            target_full_data, shadow_full_data = random_split(target_full_data, [target_full_size, shadow_full_size], generator=self.generator)

        if data_size is not None:
            if isinstance(data_size, float) and 0 < data_size <= 1.0:
                target_full_size = int(len(target_full_data) * data_size)
                shadow_full_size = int(len(shadow_full_data) * data_size)
            else:
                raise ValueError("data_size must be a float between 0 and/or including 1.")
            target_full_data, _ = random_split(target_full_data,[target_full_size, len(target_full_data) - target_full_size], generator=self.generator)
            shadow_full_data, _ = random_split(shadow_full_data,[shadow_full_size, len(shadow_full_data) - shadow_full_size], generator=self.generator)

        target_member_size = int(len(target_full_data) * 0.5)
        target_non_member_size = len(target_full_data) - target_member_size
        shadow_member_size = int(len(shadow_full_data) * 0.5)
        shadow_non_member_size = len(shadow_full_data) - shadow_member_size

        target_member_data, target_non_member_data = random_split(target_full_data,[target_member_size, target_non_member_size],generator=self.generator)
        shadow_member_data, shadow_non_member_data = random_split(shadow_full_data,[shadow_member_size, shadow_non_member_size],generator=self.generator)

        target_member_loader = DataLoader(target_member_data, batch_size=self.batch_size, shuffle=True)
        target_non_member_loader = DataLoader(target_non_member_data, batch_size=self.batch_size, shuffle=False)
        shadow_member_loader = DataLoader(shadow_member_data, batch_size=self.batch_size, shuffle=True)
        shadow_non_member_loader = DataLoader(shadow_non_member_data, batch_size=self.batch_size, shuffle=False)

        return (target_member_loader, target_non_member_loader), (shadow_member_loader, shadow_non_member_loader), shadow_dataset_name

    def _extract_purchase_dataset_(self):
        """Load the main datasets"""
        print("Loading datasets...")

        # Load main files
        transactions = pd.read_csv("./data/PURCHASE/transactions.csv.gz", compression='gzip')
        train_history = pd.read_csv("./data/PURCHASE/trainHistory.csv.gz", compression='gzip')
        test_history = pd.read_csv("./data/PURCHASE/testHistory.csv.gz", compression='gzip')
        offers = pd.read_csv("./data/PURCHASE/offers.csv.gz", compression='gzip')

        print(f"Transactions shape: {transactions.shape}")
        print(f"Train history shape: {train_history.shape}")
        print(f"Test history shape: {test_history.shape}")
        print(f"Offers shape: {offers.shape}")

        # Merge offers into train_history
        train = pd.merge(train_history, offers, on='offer', how='left')

        # Filter transactions for users in training set
        train_transactions = transactions[transactions['id'].isin(train['id'])]

        # Example aggregations: total spend, number of visits, etc.
        agg_txn = train_transactions.groupby('id').agg({
            'purchaseamount': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean'],
            'chain': 'nunique',
            'category': 'nunique'
        })

        agg_txn.columns = ['_'.join(col) for col in agg_txn.columns]
        agg_txn.reset_index(inplace=True)

        # Merge aggregated features into train set
        train = pd.merge(train, agg_txn, on='id', how='left')

