from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CentralisedDataManager:
    def __init__(self, batch_size=128, val_split=0.1):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader