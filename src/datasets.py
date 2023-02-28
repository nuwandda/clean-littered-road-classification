import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Required constants.
ROOT_DIR = '../data/Images/Images'
VALID_SPLIT = 0.2
IMAGE_SIZE = 224  # Image size of resize when applying transforms.
BATCH_SIZE = 16
NUM_WORKERS = 4  # Number of parallel processes for data preparation.


# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform


# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:  # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


def get_datasets(pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
    dataset_test = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    dataset_size = len(dataset)

    # Calculate the validation dataset size.
    valid_size = int(VALID_SPLIT * dataset_size)

    # Randomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()

    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset.classes


def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader
