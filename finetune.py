"""
This script is used to fine-tune the model on the dataset.
"""

from model.finetune import trainer, tokenizer

trainer.train()
trainer.save_model("model_finetuned")  # Save the model
tokenizer.save_pretrained("model_finetuned")  # Save the tokenizer
