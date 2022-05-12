from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional, Union
from threading import Lock

import boto3
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

import go_bindings
import logging_utils

LOGGER = logging_utils.initialize_logger(add_handler=True)

GO_LIB = go_bindings.load_go_lib("./ecClient.so")
GO_LIB.initializeVars()


class LoadTimes:
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.avg = 0
        self.sum = 0
        self.lock = Lock()

    def reset(self):
        with self.lock:
            self.num_loads = 0
            self.avg = 0
            self.sum = 0

    def update(self, val: Union[float, int]):
        with self.lock:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: Average={self.avg:.03f}\tSum={self.sum:.03f}\tCount={self.count}"


class DatasetDisk(Dataset):
    """Simulates having to load each data point from disk every call."""

    def __init__(
        self,
        filepaths: list[str],
        label_idx: int,
        dataset_name: str,
        img_transform: Optional[torchvision.transforms.Compose] = None,
    ):
        self.filepaths = np.array(filepaths)
        self.label_idx = label_idx
        self.img_transform = img_transform
        self.total_samples = 0
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        label = os.path.basename(self.filepaths[idx]).split(".")[0].split("_")[self.label_idx]
        pil_img = Image.open(self.filepaths[idx])
        if self.img_transform:
            img_tensor = self.img_transform(pil_img)
        else:
            img_tensor = F.pil_to_tensor(pil_img)
            img_tensor = img_tensor.to(torch.float32).div(255)

        self.total_samples += 1

        return img_tensor, int(label)

    def __str__(self):
        return f"{self.dataset_name}_DatasetDisk"


class MiniObjDataset(Dataset):
    def __init__(
        self,
        bucket_name: str,
        label_idx: int,
        channels: bool,
        dataset_name: str,
        img_dims: tuple[int, int, int],
        obj_size: int = 8,
        img_transform: Optional[torchvision.transforms.Compose] = None,
    ):
        self.label_idx = label_idx
        self.channels = channels
        self.s3_client = boto3.client("s3")
        self.bucket_name = bucket_name
        paginator = self.s3_client.get_paginator("list_objects_v2")
        filenames = []
        labels = []
        for page in paginator.paginate(Bucket=bucket_name):
            for content in page.get("Contents"):
                filenames.append(content["Key"])
                labels.append(int(content["Key"].split(".")[0].split("_")[self.label_idx]))

        self.object_size = obj_size
        # Chunk the filenames into objects of size self.object_size where self.object_size is the
        # number of images.
        multiple_len = len(filenames) - (len(filenames) % self.object_size)
        filenames_arr = np.array(filenames[:multiple_len])
        assert len(filenames_arr) % self.object_size == 0
        labels_arr = np.array(labels[:multiple_len])

        # Needs to be a numpy array to avoid memory leaks with multiprocessing:
        #       https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        self.chunked_fpaths = np.array(np.split(filenames_arr, (len(filenames_arr) // self.object_size)))
        self.chunked_labels = np.array(np.split(labels_arr, (len(labels_arr) // self.object_size)))

        self.base_keyname = f"{dataset_name}-{self.object_size}-"
        self.img_dims = img_dims
        self.data_type = np.uint8
        self.labels = np.ones(self.chunked_labels.shape, dtype=self.data_type)
        self.chunk_byte_lengths = np.ones(len(self.chunked_fpaths))
        self.total_samples = 0
        self.img_transform = img_transform

    def __len__(self):
        return len(self.chunked_fpaths)

    def __getitem__(self, idx: int):
        num_samples = len(self.chunked_fpaths[idx])
        key = f"{self.base_keyname}-{idx:05d}1"
        self.data_shape = (num_samples, *self.img_dims)

        try:
            bytes_arr = go_bindings.get_array_from_cache(GO_LIB, key, self.chunk_byte_lengths[idx])
            images = self.convert_bytes_to_images(bytes_arr)
            labels = torch.tensor(self.labels[idx])

        except KeyError:
            images, labels = self.get_s3_threaded(idx)
            self.labels[idx] = np.array(labels, dtype=self.data_type)
            self.chunk_byte_lengths[idx] = len(images)
            go_bindings.set_array_in_cache(GO_LIB, key, images)
            images = self.convert_bytes_to_images(images)

        if self.img_transform:
            images = self.img_transform(images.div(255))
        data = (images, labels)
        self.total_samples += num_samples
        return data

    def get_s3_threaded(self, idx: int):
        fpaths = self.chunked_fpaths[idx]
        # Returns 1-D tensor with number of labels
        labels = torch.tensor(self.chunked_labels[idx])
        byte_arr = b""
        with ThreadPoolExecutor(len(fpaths)) as executor:
            futures = [executor.submit(self.load_image, f) for f in fpaths]
            # Returns tensor of shape [object_size, num_channels, H, W]
            for future in as_completed(futures):
                byte_arr = self.convert_to_bytes(future.result(), byte_arr)

        return byte_arr, labels

    def load_image(self, s3_prefix: str) -> torch.Tensor:
        s3_png = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_prefix)
        return s3_png["Body"].read()

    def convert_to_bytes(self, inp_bytes: bytes, prev_bytes: bytes = None):
        """Combines inp_bytes into an existing byte array, if entered. Otherwise creates a new byte array.

        The first character (bytes_len) is the length of the bytes representing the length of the size
        of the bytes array. bytes_size is the size of the bytes array. This helps to break apart long
        byte arrays.
        """
        bytes_len = str(len(inp_bytes)).encode()
        bytes_size = str(len(bytes_len)).encode()

        converted_bytes = bytes_size + bytes_len + inp_bytes

        if prev_bytes:
            return prev_bytes + converted_bytes
        return converted_bytes

    def convert_bytes_to_images(self, byte_arr: bytes):
        tensor_images = torch.ones(self.data_shape)
        for idx in range(self.data_shape[0]):
            ex_bytes, byte_arr = self.extract_bytes(byte_arr)
            pil_img = Image.open(BytesIO(ex_bytes))
            tensor_images[idx] = F.pil_to_tensor(pil_img)

        return tensor_images

    def extract_bytes(self, inp_bytes: bytes) -> tuple[bytes, Optional[bytes]]:
        """Uses the byte length information used in `convert_to_bytes` to split the single byte
        array into the byte arrays for each image.
        """
        byte_len = int(inp_bytes[0:1])
        size_bytes = int(inp_bytes[1:byte_len + 1])
        sub_img = inp_bytes[1 + byte_len:size_bytes + 1 + byte_len]
        removed_bytes = inp_bytes[size_bytes + 1 + byte_len:]
        if removed_bytes:
            return sub_img, inp_bytes[size_bytes + 1 + byte_len:]
        return sub_img, None

    def set_in_cache(self, idx: int):
        key = f"{self.base_keyname}-{idx:05d}1"
        images, labels = self.get_s3_threaded(idx)
        self.labels[idx] = np.array(labels, dtype=self.data_type)
        self.chunk_byte_lengths[idx] = len(images)
        go_bindings.set_array_in_cache(GO_LIB, key, images)

    def initial_set_all_data(self):
        idxs = list(range(len(self.chunked_fpaths)))
        LOGGER.info("Loading data into InfiniCache in parallel")

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.set_in_cache, idx) for idx in idxs]
            _ = [future.result() for future in as_completed(futures)]
            LOGGER.info("DONE with initial SET into InfiniCache")
        end_time = time.time()
        time_taken = end_time - start_time

        LOGGER.info(
            "Finished Setting Data in InfiniCache. Total load time for %d samples is %.3f sec.",
            self.total_samples,
            time_taken,
        )

    def __str__(self):
        return f"{self.dataset_name}_MiniObjDataset"
