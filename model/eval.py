"""
This script is used to evaluate the model on the test set.
"""
import torch
from transformers import TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer

import dataset.llama3_dataset

from model import utils


data = dataset.llama3_dataset.Llama3Dataset("data/").load()

t5model = AutoModelForSeq2SeqLM.from_pretrained("./model_finetuned")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
def _tokenize_function(examples):
    return tokenizer(
        examples["lyrics_no_tags"],
        text_target=examples["lyrics"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

ds = data.map(_tokenize_function)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=t5model)
utils.TOKENIZER = tokenizer

BATCH_SIZE = 8

training_args = TrainingArguments(
    output_dir="eval",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
)

evaluator = Trainer(
    model=t5model,
    args=training_args,
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=utils.compute_metrics,
)
