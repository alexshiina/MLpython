"""
Contains functions for creating PyTorch DataLoaders. Inspired by mrdbourke PyTorch tutorial.
"""
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pathlib
import os
from torch.utils.data import DataLoader
NUM_WORKERS = os.cpu_count()
def dataloader(train_dir: str,
               test_dir:str,
               transform = transforms.Compose,
               batch_size = int,
               num_workers: int = NUM_WORKERS):
    """Creates dataloaders.

    Takes in a training and testing directory paths, turns them
    into Pytorch Datasets and then into PyTorch Dataloaders.

    Args:
          train_dir: Path to training directory.
          test_dir: Path to testing directory.
          transform: torchvision transforms to perform on training and testing data.
          batch_size: Number of samples per batch in each of the DataLoaders.
          num_workers: An integer number of workers per Dataloader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.

    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
    """
    #making datasets
    train_data = datasets.ImageFolder(train_dir,transform = transform)
    test_data = datasets.ImageFolder(test_dir,transform = transform)

    #getting class names
    class_names = train_data.classes

    #data into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size= batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    return train_dataloader,test_dataloader,class_names
