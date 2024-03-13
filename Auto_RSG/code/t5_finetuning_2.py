import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os


# Load and preprocess the data
class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_file, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        for item in data:
            question = item['question']['stem']
            # context_parts = [item['image_caption'], str(item['relation'])]
            # context = " ".join(part if isinstance(part, str) else " ".join(part) for part in context_parts)

            answer = item['question']['choices']['label']

            # Prepare the text input for T5
            input_text = f"question: {question} \n context: {item['image_caption']} \n relation:{str(item['relation'])}"
            target_text = answer

            # Tokenize input and target texts
            input_tokenized = tokenizer.encode_plus(input_text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
            target_tokenized = tokenizer.encode_plus(target_text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

            self.inputs.append(input_tokenized)
            self.targets.append(target_tokenized)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]['input_ids'].squeeze()
        attention_mask = self.inputs[idx]['attention_mask'].squeeze()
        target_ids = self.targets[idx]['input_ids'].squeeze()
        target_attention_mask = self.targets[idx]['attention_mask'].squeeze()

        return input_ids, attention_mask, target_ids, target_attention_mask

# Parameters
model_name = 't5-small'
batch_size = 64
learning_rate = 5e-5
epochs = 50
max_length = 512

# Model, Tokenizer, and DataLoader setup
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data_file = '/data2/KJE/GQA/statement/testdev_balanced_questions_promptcap_relation_graph_retrival4.statement.jsonl'  # Update this to your dataset file path
dataset = CustomDataset(tokenizer, data_file, max_length=max_length)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

# Training loop
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, target_ids, target_attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids, decoder_attention_mask=target_attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

from nltk.translate.bleu_score import sentence_bleu

# Validation loop (simplified)
model.eval()
total = 0
correct = 0
bleu_score_accumulator = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, target_ids, target_attention_mask = [b.to(device) for b in batch]
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        # Decode the generated ids and the target ids
        generated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in outputs]
        target_texts = [tokenizer.decode(target_id, skip_special_tokens=True) for target_id in target_ids]

        # Update total and correct counts
        total += len(batch[0])
        for generated_text, target_text in zip(generated_texts, target_texts):
            if generated_text.strip().lower() == target_text.strip().lower():
                correct += 1
            # Calculate BLEU score as well for a better evaluation metric
            reference = [target_text.strip().lower().split()]
            candidate = generated_text.strip().lower().split()
            bleu_score_accumulator += sentence_bleu(reference, candidate,
                                                    weights=(0.5, 0.5))  # Using BLEU-2 for simplicity

    accuracy = (correct / total) * 100
    average_bleu_score = bleu_score_accumulator / total

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Average BLEU-2 Score: {average_bleu_score:.4f}")