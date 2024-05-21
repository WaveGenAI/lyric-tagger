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
A module that contains the Flant5 dataset class.
"""

import json
import os
import typing

import torch
from random_word import RandomWords
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import enums
from dataset.dataset import LyricsDataset


def _setup_model() -> typing.Tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=config
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model.generation_config.pad_token_ids = tokenizer.pad_token_id

    return model, tokenizer


class Llama3Dataset(LyricsDataset):
    """A class that represents the dataset that will be generated by the llama3 model.

    Args:
        dataset (dataset.dataset.LyricsDataset): The dataset class from the dataset module.
    """

    PROMPT = """
    Generate the lyric of a song with specific tags indicating the different sections of the song. The tags that you are allowed to use are the following:
    {TAGS}
    
    The N indicates the number of the section. For example, [CHORUS 1] indicates the first chorus of the song. 
    Don't include any other text that the lyrics and the tags. Don't use any other tags than the ones mentioned.
    The lyrics should be generated based on the word "{WORD}" and have to contain the word at the 5th word of the lyrics.
    """

    def __init__(self, path: str) -> None:
        """The constructor for the Flant5 dataset.

        Args:
            path (str): The path where the lyrics are stored.
        """

        self._path = path
        self._tags = enums.Tag
        self._word_generator = RandomWords()

        super().__init__(path)

    def clean_lyrics(self, file_id: list = None) -> None:
        """Function that cleans the lyrics from the dataset.

        Args:
            file_id (list, optional): the list of lyrics to delete, if empty delete everything. Defaults to [].
        """

        if file_id is None:
            for file in os.listdir(self._path):
                if file.endswith(".txt"):
                    os.remove(os.path.join(self._path, file))
        else:
            for idx in file_id:
                os.remove(os.path.join(self._path, f"{idx}.txt"))

    def generate_dataset(self, nb_gen: int = 100) -> None:
        """Function that generates the dataset.

        Args:
            nb_gen (int, optional): the number of lyric generated. Defaults to 100.
        """

        model, tokenizer = _setup_model()

        # include the tags in the prompt from the enums
        tags = ""
        for tag in self._tags:
            tags += f"[{tag.value} N]\n"

        # generate the number of lyrics specified
        for i in range(nb_gen):
            # add word to obtain different lyrics each time
            word = " ".join(self._word_generator.get_random_word() for _ in range(1))

            messages = [
                {
                    "role": "system",
                    "content": "You are a chatbot that generates lyrics.",
                },
                {
                    "role": "user",
                    "content": self.PROMPT.format(TAGS=tags, WORD=word),
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                input_ids,
                max_length=500,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.9,
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

            # remove the prompt and the special tokens
            generated_text = (
                generated_text.replace(prompt, "")
                .replace("<|begin_of_text|>", "")
                .replace("<|eot_id|>", "")
            )

            with open(
                os.path.join(self._path, f"{i}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(generated_text)
