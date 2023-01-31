import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch
from transformers import AutoTokenizer, AutoModel
import psutil
from model.model import Model
from datasets import load_from_disk, Value

# don't change unless you know why
SEED = 42

classes = 2
d_model = 64
s_len_1 = 768
s_len_2 = 1900
n = 3
heads = 2
dropout = 0.1
batch_size = 4

def map_label(e):
    if e['label'] == 1.0:
        e['label'] = np.array([1.0, 0.0])
    else:
        e['label'] = np.array([0.0, 1.0])
    return e

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(classes, d_model, s_len_1, s_len_2, n, heads, dropout, device)
model.to(device)

dataset = load_from_disk('dataset')
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
#dataset = dataset.map(map_label, batched=False)
dataset = dataset.cast_column('label', Value("int32"))
dataset = dataset.map(lambda e: tokenizer(e['Canonical SMILE'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'embedding', 'label'])

dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
train = dataset['train']
test = dataset['test']

batch_sampler = BatchSampler(RandomSampler(train), batch_size=batch_size, drop_last=False)
train_loader = DataLoader(train, batch_sampler=batch_sampler)

test_loader = DataLoader(test, batch_size=batch_size)




criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)

losses = pd.DataFrame()
epochs = 2
patience = 5
curr_patience = 5
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    model.train()  # Optional when not using Model Specific layer
    timep = time.process_time()
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        target = model(batch, device)
        loss = criterion(target, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if iter % 1000 == 0:
            print(f'Loss for iteration {iter} was {loss.item()}')
        if iter % 100000 == 0:
            torch.save(model.state_dict(), 'backup_model.pth')
    timep = time.process_time() - timep
    print(f"epoch took {timep//60} minutes and {round(timep)%60} seconds")
    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for batch in test_loader:
        target = model(batch, device)
        loss = criterion(target, batch['label'].to(device))
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