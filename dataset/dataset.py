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

import json
import re

from torch.utils.data import Dataset

import enums
import exceptions


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
    return text


class LyricsDataset(Dataset):
    """A class that represents the dataset that contains all the lyrics.

    Args:
        Dataset (torch.utils.data.Dataset): The dataset class from PyTorch.
    """

    def __init__(self, json_path: str = "./data/lyrics.json") -> None:
        super().__init__()
        self._data = None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except json.JSONDecodeError as e:
            raise exceptions.NotValidJsonFile(
                f"{json_path} is not a valid JSON file."
            ) from e

    def __len__(self) -> int:
        if self._data is None:
            raise exceptions.NotLoadedJsonFile("JSON file is not loaded.")

        return len(self._data)

    def __getitem__(self, idx: int) -> str:
        if self._data is None:
            raise exceptions.NotLoadedJsonFile("JSON file is not loaded.")

        lyrics = self._data[str(idx)]
        return lyrics, _remove_tags(lyrics)
