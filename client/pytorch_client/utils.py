from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms
import os


def split_cifar_data(cifar_data_dir: str, test_fnames_path: str) -> list[str]:
    """
    Expects all files to be in a single directory with the same filenames as in the S3 bucket.
    """
    dataset_path = Path(cifar_data_dir)
    filenames = list(dataset_path.rglob("*.png"))
    filenames.extend(list(dataset_path.rglob("*.jpg")))
    with open(test_fnames_path) as f:
        test_fnames = set([fname.strip() for fname in f.readlines()])
    filestubs = set(map(lambda x: x.name, filenames))
    train_filenames = list(filestubs.difference(test_fnames))
    train_filenames = sorted(train_filenames, key=lambda filename: filename)
    train_filenames = list(map(lambda x: os.path.join(cifar_data_dir, x), train_filenames))
    test_filenames = sorted(test_fnames, key=lambda filename: filename)
    test_filenames = list(map(lambda x: os.path.join(cifar_data_dir, x), test_filenames))

    return train_filenames, test_filenames


def normalize_image(channels: bool):
    if not channels:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485), (0.229)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transform


def cifar_transforms_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def infinicache_collate(batch):
    """Custom collate function to work with the mini-objects
    """
    images, labels = zip(*batch)
    new_images = torch.cat(images, 0)
    new_labels = torch.cat(labels, 0)
    return new_images, new_labels
