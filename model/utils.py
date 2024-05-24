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
"""
A module that contains the utility functions for the model training process.
"""
import evaluate
import numpy as np
import torch

metric = evaluate.load("sacrebleu")
TOKENIZER = None


def postprocess_text(preds, labels):
    """
    Postprocess the text for the evaluation.
    Args:
        preds: The predictions to evaluate.
        labels: The labels as reference.

    Returns: The postprocessed predictions and labels.

    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    """
    Compute the metrics for the evaluation.
    Args:
        eval_preds: The evaluation predictions.

    Returns:

    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=-1)
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != TOKENIZER.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


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
