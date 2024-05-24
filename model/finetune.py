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
A module that contain the script to finetune the model.
"""
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

import dataset.llama3_dataset
from model import utils
from model. patch import patch

patch()

data = dataset.llama3_dataset.Llama3Dataset("data/").load()

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1
)
t5model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
special_tokens_to_add = []
for i in range(1, 5):
    special_tokens_to_add.append(f"[CHORUS {i}]")
    special_tokens_to_add.append(f"[VERSE {i}]")
special_tokens_to_add.append("[BRIDGE 1]")
special_tokens_to_add.append("\n")
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
t5model.resize_token_embeddings(len(tokenizer))
# t5model = get_peft_model(t5model, peft_config)
# t5model.print_trainable_parameters()


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
    output_dir="training",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=6,
    learning_rate=8e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
    logging_steps=50,
    optim="adamw_8bit",
    save_strategy="epoch",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=t5model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    # compute_metrics=model.utils.compute_metrics,
)
