"""
A script that generates the lyrics for the dataset.
"""

from dataset.llama3_dataset import Llama3Dataset

dataset = Llama3Dataset("data/")
# dataset.generate_dataset()
data = dataset.load()

for d in data["test"]:
    print(d)


# dataset.clean_lyrics()

# test section
#
# try:
#     for data in dataset:
#         lyrics = data[0]
#         lyrics_no_tags = data[1]
# except KeyError as e:
#     pass
