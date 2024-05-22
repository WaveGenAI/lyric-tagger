"""
This script is used to fine-tune the model on the dataset.
"""

from model.finetune import trainer

trainer.train()
trainer.save_model("model_fintuned")  # Save the model
