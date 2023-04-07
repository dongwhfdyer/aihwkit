import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('./model_cache/bert-base-uncased')
model = AutoModel.from_pretrained('./model_cache/bert-base-uncased')

# Tokenize input text
input_text = "Hello, how are you doing today?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate output
with torch.no_grad():
    output = model(input_ids)
    print("--------------------------------------------------")

# Load the Wikipedia dataset
from datasets import load_dataset
dataset = load_dataset('wikipedia', '20220301.en', split='train[:1%]')

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)

# Run inference on the tokenized dataset using the pre-trained model
with torch.no_grad():
    output = model(**tokenized_dataset)
    print("--------------------------------------------------")
