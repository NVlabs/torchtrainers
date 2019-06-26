#
# Copyright (c) 2013-2019 Thomas Breuel. All rights reserved.
# This file is part of torchtrainers (see unknown).
# See the LICENSE file for licensing terms (BSD-style).
#
"""Training-related part of the Keras engine.
"""

import sys
import numpy as np
import torch
import time
import random
import PIL

def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    :param sample: list of values
    :param transformers: list of functions

    """
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    ntransformers = len(transformers)
    for i in range(len(sample)):
        f = transformers[i%ntransformers]
        if f is not None:
            result[i] = f(sample[i])
    return result

def onehot(nclasses, dtype=torch.float32):
    """Compute one-hot encoding (numpy+torch)."""
    def f(ys):
        if isinstance(ys, list):
            ys = np.ndarray(ys, dtype="int64")
        if isinstance(ys, np.ndarray):
            result = np.zeros((len(ys), nclasses))
            result[np.arange(len(ys)), ys] = 1
            return result
        elif isinstance(ys, torch.Tensor):
            result = torch.zeros_like(ys, dtype=dtype)
            result.scatter(1, ys, 1)
            return result
        else:
            raise ValueError("unknown dtype", ys.dtype)
    return f

def convert_uint8(xs, dtype=torch.float32):
    if isinstance(xs, torch.Tensor) and xs.dtype==torch.uint8:
        xs = xs.type(dtype) / 255.0
    elif isinstance(xs, np.ndarray) and xs.dtype==np.uint8:
        xs = xs.type(dtype) / 255.0
    return xs

class QuickLoader(object):
    def __init__(self, xs, ys,
                 batch_size=64,
                 epochs=1,
                 size=None,
                 shuffle=True,
                 device="cpu",
                 handle_images=True,
                 uint8_to_float=True,
                 batch_pre=None,
                 batch_convert=None,
                 batch_transforms=None):
        self.xs = xs
        self.ys = ys
        self.size = len(self.xs) if size is None else size
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.handle_images = handle_images
        self.uint8_to_float = uint8_to_float
        self.batch_pre = batch_pre
        self.batch_transforms = batch_transforms
        self.batch_convert = batch_convert
    def __iter__(self):
        for epoch in range(self.epochs):
            indexes = list(range(0, len(self.xs), self.batch_size))
            if self.shuffle:
                random.shuffle(indexes)
            for index in indexes:
                xs = self.xs[index:index+self.batch_size]
                ys = self.ys[index:index+self.batch_size]
                if self.uint8_to_float:
                    xs = convert_uint8(xs)
                    ys = convert_uint8(ys)
                if self.batch_pre is not None:
                    xs, ys = transform_with([xs, ys], self.batch_pre)
                if self.batch_transforms is not None:
                    xs, ys = transform_with([xs, ys], self.batch_transforms)
                if self.batch_convert is not None:
                    xs, ys = transform_with([xs, ys], self.batch_convert)
                yield xs, ys
    def __len__(self):
        return (len(self.xs) + self.batch_size - 1) // self.batch_size
