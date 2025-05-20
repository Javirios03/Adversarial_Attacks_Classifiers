import pickle
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_data(file) -> dict:
    print("Loading CIFAR-10 data from: ", file)
    data = unpickle(file)
    return data


def get_cifar10_loaders(data_dir="/Git/data/CIFAR_10", batch_size: int = 128) -> tuple:
    '''
    Get the CIFAR-10 data loaders

    Parameters:
    data: dict
        A dictionary containing the CIFAR-10 data (5 training batches and 1 test batch)
    batch_size: int
        The batch size for the data loaders

    Returns:
    train_loader: torch.utils.data.DataLoader
        The training data loader
    test_loader: torch.utils.data.DataLoader
        The test data loader
    '''
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# def main() -> dict:
#     '''
#     Load CIFAR-10 data from the files

#     Returns:
#     data: dict
#         A dictionary containing the CIFAR-10 data (5 training batches and 1 test batch)
#     '''
#     path = "Git/data/CIFAR_10/cifar-10-batches-py/"
#     files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
#     data = {}
#     for file in files:
#         data[file] = load_cifar10_data(path + file)
#     return data
