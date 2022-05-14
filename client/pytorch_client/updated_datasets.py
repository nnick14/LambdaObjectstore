from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from threading import Lock
from functools import partial
from random import randint

import boto3
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

import go_bindings 
import logging_utils

LOGGER = logging_utils.initialize_logger()
DATALOG = logging_utils.get_logger("datalog")
INITIALIZE_WORKERS = 10

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
        s3_bucket: str = "",
    ):
        self.dataset_name = dataset_name
        
        if len(filepaths) == 1:
            localpath = Path(filepaths[0])

            # Download
            if s3_bucket != "":
                self.s3_client = boto3.client("s3")    
                self.download_from_s3(s3_bucket, localpath.absolute())  # We probably don't need this since the AWS CLI is faster\
            
            # Expand
            filepaths = list(localpath.rglob("*.png"))
            filepaths.extend(list(localpath.rglob("*.jpg")))
            filepaths = list(map(lambda x: str(x), filepaths))

        self.filepaths = np.array(filepaths)
        self.label_idx = label_idx
        self.img_transform = img_transform
        self.total_samples = 0

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

    def download_from_s3(self, s3_path: str, local_path: str):
        """
        First need to download all images from S3 to Disk to use for training.
        """
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        filenames = []
        for page in paginator.paginate(Bucket=s3_path):
            for content in page.get("Contents"):
                filenames.append(content["Key"])
        partial_dl = partial(self.download_file, local_path, s3_path)
        LOGGER.info("Downloading data from S3 to Disk")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(partial_dl, fname) for fname in filenames]
            _ = [future.result() for future in as_completed(futures)]
        LOGGER.info("Download is complete")

    def download_file(self, output_dir: str, bucket_name: str, file_name: str):
        self.s3_client.download_file(bucket_name, file_name, f"{output_dir}/{file_name}")


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
        self.dataset_name = dataset_name

        LOGGER.info("Initializing dataset {} from s3".format(dataset_name))
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
        self.metas = np.full(len(self.chunked_labels), None)
        self.total_samples = 0
        self.img_transform = img_transform

        LOGGER.info("Dataset {} initialized".format(dataset_name))

    def __len__(self):
        return len(self.chunked_fpaths)

    def __getitem__(self, idx: int):
        num_samples = len(self.chunked_fpaths[idx])
        key = f"{self.base_keyname}-{idx:05d}"
        self.data_shape = (num_samples, *self.img_dims)

        try:
            meta = self.metas[idx]
            if meta is None:
                raise KeyError("Key not set")
            bytes = go_bindings.get_array_from_cache(go_bindings.GO_LIB, key, meta[0])
            # images = torch.tensor(np_arr).reshape(self.data_shape)
            np_arr = self.unwrap(bytes, meta)
            labels = self.labels[idx]
            np_arr, labels = self.shuffle(np_arr, np.copy(labels))

            # Keep load_images in try block, so we may reset it if necessary
            images = torch.stack(list(map(lambda x: self.load_image(x), np_arr)))
        except Exception as e:
            LOGGER.debug("{} Resetting image {} due to {}".format(idx, key, e))
            np_arr, labels = self.set_in_cache(idx)
            np_arr, labels = self.shuffle(np_arr, np.copy(labels))
            images = torch.stack(list(map(lambda x: self.load_image(x), np_arr)))

        data = (images, torch.tensor(labels))
        self.total_samples += num_samples
        return data

    def shuffle(self, arr1, arr2):
        """
        Permute the arrays synchornizely.
        """
        if len(arr1) != len(arr2):
            raise ValueError("Arrays to be shuffled must be the same length")
        for i in range(len(arr1)-1,0,-1):
            j = randint(0,i)
            arr1[i], arr1[j] = arr1[j], arr1[i]
            arr2[i], arr2[j] = arr2[j], arr2[i]
        
        return arr1, arr2

    def get_s3_threaded(self, idx: int):
        fpaths = self.chunked_fpaths[idx]
        # Returns 1-D tensor with number of labels
        labels = torch.tensor(self.chunked_labels[idx])
        with ThreadPoolExecutor(len(fpaths)) as executor:
            futures = [executor.submit(self.load_s3, f) for f in fpaths]
            # Returns tensor of shape [object_size, num_channels, H, W]
            # results = torch.stack([future.result() for future in as_completed(futures)])
            results = [future.result() for future in as_completed(futures)]
        return results, labels

    def load_image(self, img_bytes: bytes) -> torch.Tensor:
        pil_img = Image.open(BytesIO(img_bytes))
        if self.img_transform:
            img_tensor = self.img_transform(pil_img)
        else:
            img_tensor = F.pil_to_tensor(pil_img)
            img_tensor = img_tensor.to(torch.float32).div(255)
        return img_tensor

    def load_s3(self, s3_prefix: str) -> any:
        s3_png = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_prefix)
        img_bytes = s3_png["Body"].read()
        return img_bytes
        # pil_img = Image.open(BytesIO(img_bytes))
        # img_tensor = F.pil_to_tensor(pil_img)
        # return img_tensor

    def set_in_cache(self, idx: int):
        key = f"{self.base_keyname}-{idx:05d}"
        # LOGGER.debug("{}. setting images {}".format(idx, key))
        # bytes_loaded = 0
        img_bytes, labels = self.get_s3_threaded(idx)
        # for i in images:
        #     bytes_loaded += len(i)
        arr = np.array(img_bytes)
        self.labels[idx] = np.array(labels, dtype=self.data_type)
        # tbs = np.array(images).astype(self.data_type)
        tbs = self.wrap(arr)
        self.metas[idx] = [len(tbs), arr.dtype]
        # LOGGER.debug("Setting in cache: {} images, read {} bytes, setting {} bytes: {}".format(len(images), bytes_loaded, len(tbs), list(map(lambda x: len(x), arr))))
        go_bindings.set_array_in_cache(go_bindings.GO_LIB, key, tbs)
        return arr, self.labels[idx]

    def set_in_cache_threaded(self, idx: int):
        self.initial_progress[idx] = 2
        _ = self.set_in_cache(idx)
        self.initial_progress[idx] = 1
        return idx

    def track_threads(self, future):
        ret = future.result()
        self.initial_finished += 1
        start = ret - INITIALIZE_WORKERS
        if start < 0:
            start = 0
        end = ret + INITIALIZE_WORKERS
        if end > len(self.chunked_fpaths):
            end = len(self.chunked_fpaths)
        lookback = list(filter(lambda x: self.initial_progress[x] != 1, range(start, ret)))
        lookahead = list(filter(lambda x: self.initial_progress[x] == 2, range(ret, end)))
        LOGGER.debug("Initiated {}/{} objects, status {}:{}:{}".format(
            self.initial_finished,
            len(self.initial_progress),
            list(map(lambda x: -x if self.initial_progress[x] == 0 else x, lookback)),
            ret, 
            lookahead
        ))
        return ret

    def wrap(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()

    def unwrap(self, bytes_arr: bytes, meta: any) -> np.ndarray:
        arr = np.frombuffer(bytes_arr, dtype=meta[1])
        # LOGGER.debug("Unwraped {} bytes: {}".format(len(bytes_arr), list(map(lambda x: len(x), arr))))
        # array from buffer is readonly. We will need to shuffle the array, so return a copy.
        return np.copy(arr)

    def initial_set_all_data(self):
        idxs = list(range(len(self.chunked_fpaths)))
        LOGGER.info("Loading {} into InfiniCache in parallel".format(self.dataset_name))

        start_time = time.time()
        self.initial_progress = np.full(len(idxs), 0)
        self.initial_finished = 0
        with ThreadPoolExecutor(max_workers=INITIALIZE_WORKERS) as executor:
            futures = [executor.submit(self.set_in_cache_threaded, idx) for idx in idxs]
            _ = [self.track_threads(future) for future in as_completed(futures)]
            LOGGER.info("DONE with initial SET into InfiniCache")
        end_time = time.time()
        time_taken = end_time - start_time

        LOGGER.info(
            "Finished Setting Data in InfiniCache. Total load time for %d samples is %.3f sec.",
            self.total_samples,
            time_taken,
        )
        return time_taken, self.total_samples

    def __str__(self):
        return f"{self.dataset_name}_MiniObjDataset"
