# lyric-tagger
The codebase for finetuning flant5 model to include tag in a lyric.

## Dataset

The dataset should be in the following structure:

```
data/
  1.txt
  2.txt
  ...
  lyrics.json
```

## Usage

To train the model, run the following command:

```
python finetune.py
```

To generate the dataset, run the following command:

```
python generate_lyric.py
```

## Setup

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
