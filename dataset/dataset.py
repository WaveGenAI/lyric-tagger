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
Module that contains the dataset class for the lyrics.
"""

import re
import random

import datasets
import enums


def _remove_tags(text: str) -> str:
    """Function that removes all the tags from the lyrics.

    Args:
    text (str): The lyrics text.

    Returns:
    str: The lyrics text without the tags.
    """

    for tag in enums.Tag:
        text = re.sub(rf"\[{tag.value} [0-9]*\]|\[{tag.value}\]", "\n", text)
        text = text.replace("\n\n\n", "\n")
        if random.random() <= 0.5:
            text = text.replace("\n\n", "").replace("\n", "")
    return text


class LyricsDataset:
    """A class that represents the dataset that contains all the lyrics."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def load(self) -> datasets.dataset_dict.DatasetDict:
        """Method that loads the dataset.

        Returns:
            datasets.dataset_dict.DatasetDict: The dataset.
        """

        dataset = datasets.load_dataset(
            "text", data_dir=self._path, sample_by="document", split="train"
        )

        dataset = dataset.map(
            lambda x: {
                "lyrics_no_tags": _remove_tags(x["text"]),
                "lyrics": x["text"],
            }
        )

        test_size = min(int(len(dataset) * 0.2), 2000)
        dataset = dataset.train_test_split(test_size=test_size)

        return dataset
