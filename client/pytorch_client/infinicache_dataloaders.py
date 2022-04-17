"""
This has been adjusted so that it works for images of any shape. MNIST has images with dimension
(1, 28, 28) (784), while CIFAR-10 has images with dimension (3, 32, 32).

The CIFAR-10 images are only slightly larger than the MNIST images 32x32x3 (3072 bytes)
vs 28x28x1 (784 bytes), respectively. After running some tests, the load times were similar,
so we decided to use a subset of the ImageNet dataset instead. We selected 10 classes of images
from the dataset (tench, English springer, cassette player, chain saw, church, French horn,
garbage truck, gas pump, golf ball, parachute.
This amounted to about 900 examples each for 8,937 total images we and resized each image to
256x256x3 (196,608 bytes).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import Union
from io import BytesIO
from pathlib import Path
from typing import Callable

import boto3
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils import collate

import go_bindings
import logging_utils

LOGGER = logging_utils.initialize_logger(add_handler=True)

GO_LIB = go_bindings.load_go_lib("./ecClient.so")
GO_LIB.initializeVars()


class DatasetDisk(Dataset):
    """Simulates having to load each data point from disk every call."""

    def __init__(self, data_path: str, label_idx: int):
        dataset_path = Path(data_path)
        filenames = list(dataset_path.rglob("*.png"))
        filenames.extend(list(dataset_path.rglob("*.jpeg")))
        self.filepaths = sorted(filenames, key=lambda filename: filename.stem)
        self.label_idx = label_idx

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        label = self.filepaths[idx].stem.split("_")[self.label_idx]
        img = torchvision.io.read_image(str(self.filepaths[idx]))
        return img.to(torch.float32), int(label)


class DatasetS3(Dataset):
    """Simulates having to load each data point from S3 every call."""

    def __init__(self, bucket_name: str, label_idx: int, channels: bool):
        self.label_idx = label_idx
        self.channels = channels
        self.s3_client = boto3.client("s3")
        self.bucket_name = bucket_name
        paginator = self.s3_client.get_paginator("list_objects_v2")
        filenames = []
        for page in paginator.paginate(Bucket=bucket_name):
            for content in page.get("Contents"):
                filenames.append(Path(content["Key"]))
        self.filepaths = sorted(filenames, key=lambda filename: filename.stem)
        if channels:
            self.transform_array = self.transform_image_channels
        else:
            self.transform_array = self.transform_image_no_chan

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        label = self.filepaths[idx].stem.split("_")[self.label_idx]
        s3_png = self.s3_client.get_object(Bucket=self.bucket_name, Key=str(self.filepaths[idx]))
        img_bytes = s3_png["Body"].read()

        img_tensor = self.transform_array(img_bytes)
        return img_tensor.to(torch.float32), int(label)

    @staticmethod
    def transform_image_no_chan(img_bytes: bytes) -> torch.Tensor:
        img = np.array(Image.open(BytesIO(img_bytes)))
        return torch.from_numpy(img)

    @staticmethod
    def transform_image_channels(img_bytes: bytes) -> torch.Tensor:
        img = np.array(Image.open(BytesIO(img_bytes))).transpose(2, 0, 1)  # Need channels first
        return torch.from_numpy(img)


class BaseDataLoader(ABC):
    def __init__(
        self,
        dataset: Union[DatasetS3, DatasetDisk],
        dataset_name: str,
        img_dims: tuple[int, int, int],
        image_dtype: go_bindings.NumpyDtype,
        batch_size: int,
        collate_fn: Callable,
    ):
        self.index = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.img_dims = img_dims
        self.data_type = image_dtype
        self.labels_cache = {}
        self.load_times = []
        self.total_samples = 0
        self.dataset_name = dataset_name

    def __iter__(self):
        self.index = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item

    def __len__(self):
        return len(self.dataset) // self.batch_size


class DiskLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: DatasetDisk,
        dataset_name: str,
        img_dims: tuple[int, int, int],
        image_dtype: go_bindings.NumpyDtype = np.uint8,
        batch_size: int = 64,
        collate_fn: Callable = collate.default_collate,
    ):
        super().__init__(dataset, dataset_name, img_dims, image_dtype, batch_size, collate_fn)

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        start_time = time.time()
        data = self.collate_fn([self.get() for _ in range(batch_size)])

        end_time = time.time()
        time_taken = end_time - start_time
        self.total_samples += batch_size
        self.load_times.append(time_taken)
        return data

    def __str__(self):
        return "DiskDataset"


class S3Loader(BaseDataLoader):
    def __init__(
        self,
        dataset: DatasetS3,
        dataset_name: str,
        img_dims: tuple[int, int, int],
        image_dtype: go_bindings.NumpyDtype = np.uint8,
        batch_size: int = 64,
        collate_fn: Callable = collate.default_collate,
    ):
        super().__init__(dataset, dataset_name, img_dims, image_dtype, batch_size, collate_fn)

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        start_time = time.time()

        results = self.get_batch_threaded(batch_size)
        images, labels = self.collate_fn(results)
        images = images.reshape(batch_size, *self.img_dims)
        data = (images, labels)

        end_time = time.time()
        time_taken = end_time - start_time
        self.total_samples += batch_size
        self.load_times.append(time_taken)
        return data

    def __str__(self):
        return "S3Dataset"

    def get_batch_threaded(self, batch_size: int):
        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.get) for _ in range(batch_size)]
            results = [future.result() for future in as_completed(futures)]
        return results


class InfiniCacheLoader(BaseDataLoader):
    """DataLoader specific to InfiniCache. Associates each batch of images with a key in the cache,
    rather than each image.
    """

    def __init__(
        self,
        dataset: DatasetS3,
        dataset_name: str,
        img_dims: tuple[int, int, int],
        image_dtype: go_bindings.NumpyDtype = np.uint8,
        batch_size: int = 64,
        collate_fn: Callable = collate.default_collate,
    ):
        super().__init__(dataset, dataset_name, img_dims, image_dtype, batch_size, collate_fn)

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        start_time = time.time()
        key = f"{self.dataset_name}_batch_test600{self.batch_size}_{self.index:05d}"
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        self.data_shape = (batch_size, *self.img_dims)

        try:
            np_arr = go_bindings.get_array_from_cache(GO_LIB, key, self.data_type, self.data_shape)
            images = np_arr.reshape(self.data_shape)
            images = torch.tensor(np_arr).reshape(self.data_shape)
            labels = self.labels_cache[key]
            self.index += self.batch_size
            data = (images.to(torch.float32), labels)

        except KeyError:
            results = self.get_batch_threaded(batch_size)
            images, labels = self.collate_fn(results)
            self.labels_cache[key] = labels
            go_bindings.set_array_in_cache(GO_LIB, key, np.array(images).astype(self.data_type))
            images = images.to(torch.float32).reshape(self.data_shape)
            data = (images, labels)
        end_time = time.time()
        time_taken = end_time - start_time
        self.total_samples += batch_size
        self.load_times.append(time_taken)

        return data

    def __str__(self):
        return "InfiniCacheDataset"

    def get_batch_threaded(self, batch_size: int):
        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.get) for _ in range(batch_size)]
            results = [future.result() for future in as_completed(futures)]
        return results
