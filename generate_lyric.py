"""
A script that generates the lyrics for the dataset.
"""

from dataset.llama3_dataset import Llama3Dataset

dataset = Llama3Dataset("data/")
dataset.clean_lyrics()
dataset.generate_dataset(10_000)

# test section
#
# try:
#     for data in dataset:
#         lyrics = data[0]
#         lyrics_no_tags = data[1]
# except KeyError as e:
#     pass
