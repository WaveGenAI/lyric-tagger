#
# Created on Wed May 15 2024
#
# The MIT License (MIT)
# Copyright (c) 2024 WaveAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import evaluate
import numpy as np
import torch

metric = evaluate.load("perplexity")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def split_dataset(dataset) -> tuple:
    """
    Split the dataset into training and validation set.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.

    Returns:
        tuple: A tuple containing the training and validation set.
    """

    test_size = min(int(0.2 * len(dataset)), 2000)
    train_size = len(dataset) - test_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_set, val_set
