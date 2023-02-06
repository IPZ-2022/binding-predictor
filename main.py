import os.path
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from datasets import load_from_disk, Value
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from transformers import AutoTokenizer

from model.model import Model

# don't change unless you know why
SEED = 42
# adjust these as necessary
classes = 2
d_model = 64
s_len_1 = 768
s_len_2 = 1900
n = 3
heads = 2
dropout = 0.1
batch_size = 16*8 # batch size * num of gpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(classes, d_model, s_len_1, s_len_2, n, heads, dropout, device)
state_fixed = OrderedDict()
state = None
# backup_model.pth is available at https://drive.google.com/file/d/1r_P4K7DaRIjldqx0oloyzYt2PXaa4I5T/view?usp=sharing
if os.path.isfile('backup_model.pth'):
    state = torch.load('backup_model.pth')
    # hot fix for weights being saved with incorrect names
    for k, v in state['model-dict'].items():
        state_fixed[k[7:]] = v
        model.load_state_dict(state_fixed)

model = DP(model) # for multiple gpus
model.to(device)

dataset = load_from_disk('dataset')
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

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
if os.path.isfile('backup_model.pth'):
    optimizer.load_state_dict(state['optim-dict'])

losses = pd.DataFrame()
epochs = 8
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
            torch.save({
                        'model-dict':model.state_dict(),
                        'step':iter,
                        'epoch':e,
                        'optim-dict':optimizer.state_dict(),
                        'loss':loss
                        }, 'backup_model.pth')
    timep = time.process_time() - timep
    print(f"epoch took {timep//60} minutes and {round(timep)%60} seconds")
    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer

    y, y_ = [], []
    for batch in test_loader:
        target = model(batch, device)
        labels = batch['label'].to(device)
        loss = criterion(target, labels)
        y_.extend(target.detach().cpu().numpy()[:, 1] > 0.5)
        y.extend(labels.detach().cpu().numpy())
        valid_loss += loss.item()
    print(f'Accuracy: {accuracy_score(y, y_)}  |  Precision {precision_score(y, y_)}  |  Recall {recall_score(y, y_)}')

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
plt.savefig('train-plot.png')