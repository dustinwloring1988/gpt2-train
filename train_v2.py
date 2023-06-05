import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset

# Define hyperparameters
batch_size = 4
learning_rate = 1e-5
num_epochs = 3

# Load the custom dataset
dataset = load_dataset('dloring1/custom_chatv_0_1')

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize and encode the dataset
def encode_dataset(examples):
    inputs = tokenizer.batch_encode_plus(
        examples['prompt'],
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = tokenizer.batch_encode_plus(
        examples['answer'],
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_dataset = dataset.map(encode_dataset, batched=True)

# Prepare the DataLoader
dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)

# Load the GPT-2 model
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)

# Define the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs = {k: torch.stack([example[k] for example in batch]) for k in batch[0] if k != 'labels'}
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.stack([example['labels'] for example in batch]).squeeze().to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save the trained model
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_tokenizer')
