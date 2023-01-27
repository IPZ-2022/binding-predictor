import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
import psutil
from dataset import get_data
from model.model import Model

classes = 2
d_model = 64
s_len_1 = 512
s_len_2 = 512
n = 3
heads = 2
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


embedder = embed.UniRepEmbedder(device=device)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def process_dataset(example):
    example['seq_embed'] = embedder.embed_many(example['Sequence'])
    return tokenizer(example['Canonical SMILE'], truncation=True, padding="max_length")


dataset = get_data()

dataset.map(process_dataset, batched=True)

ex = dataset[0]


model = Model(classes, d_model, s_len_1, s_len_2, n, heads, dropout, device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

losses = pd.DataFrame()
epochs = 2
patience = 5
curr_patience = 5
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    model.train()  # Optional when not using Model Specific layer
    for data, labels in train_loader:
        if torch.cuda.is_available():
            (x_1, x_2), labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        target = model(*data)
        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(*data)
        loss = criterion(target, labels)
        valid_loss = loss.item()

    losses = losses.append({'Train_Loss': train_loss, 'Test_Loss': valid_loss}, ignore_index=True)

    print(
        f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(test_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'best_model.pth')
        curr_patience = patience
    else:
        curr_patience -= 1
    if curr_patience == 0:
        break

losses.plot()
plt.show()