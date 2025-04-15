# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json


class TarotDataset(Dataset):
    def __init__(self, readings_file, card_names_file, tokenizer = None):
        # Load readings
        with open(readings_file, 'r') as f:
            self.readings = json.load(f)

        # Load card names
        with open(card_names_file, 'r') as f:
            card_names_dict = json.load(f)
            self.card_names = [card_names_dict[str(i)] for i in range(len(card_names_dict))]

        self.tokenizer = tokenizer
        self.reading_ids = list(self.readings.keys())

    def __len__(self):
        return len(self.reading_ids)

    def __getitem__(self, idx):
        reading_id = self.reading_ids[idx]
        reading = self.readings[reading_id]

        cards = reading['cards']
        context = reading['context']
        interpretation = reading['interpretation']

        # Convert to tensor format
        cards_tensor = torch.tensor(cards, dtype = torch.long)

        # Tokenize interpretation if tokenizer provided
        if self.tokenizer:
            interpretation_tokens = torch.tensor(self.tokenizer.encode(interpretation), dtype = torch.long)
        else:
            interpretation_tokens = torch.tensor([0])  # Placeholder

        return {
            'cards': cards_tensor,
            'context': context,
            'interpretation': interpretation_tokens
        }


def train(model, dataset, epochs = 10, batch_size = 32, lr = 0.001):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Forward pass
            logits = model(batch['cards'], batch['context'])

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch['interpretation'].view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model