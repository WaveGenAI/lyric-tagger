"""
This module tests the trained model with real data.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LYR = """Your shoulders colder than December since you turned away from me
and it's been so long I can't remember
when I wasn't your enemy like you
don't want this what I really do
I wonder why
I want to be back on your good side
because nothing I do ever seems right
Stuck in a cycle for days
By my own what to say
Get by on better days
I want on your good side
One on your good side
Nobody knows me like you're going to try
Nobody knows me like you know me
And that's why it's killing me
Because there's no one else I'd rather hold me
When I'm going through misery like you
Don't want this what I really do
And how do I?
I want to be back on your good side
Because nothing I do ever seems right
Stuck in a cycle for days
By my own what to say
Get by all better days
I want on your good side
I want to be good side
I want to be back on your good side
There's nothing I do ever seems right
Stuck in a cycle for days
By my own what to say
Get by on better days
I want to be good side
I want a good side
I want a good side
side
I want you to decide
Oh, I want you to decide
Oh, want you to decide"""

model = AutoModelForSeq2SeqLM.from_pretrained("./model_finetuned").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./model_finetuned")

input_ids = tokenizer(LYR, return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(input_ids, max_length=512, do_sample=True, temperature=0.7, top_k=50)
text = tokenizer.decode(out[0])
print(text)
