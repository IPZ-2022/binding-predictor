#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model.model import Model


# In[6]:


df_ic50 = pd.read_csv('cleared_data/IC50_protein_compound_pair.tsv', delimiter='\t')
df_uniprot = pd.read_csv('cleared_data/dpid_dom.tsv', delimiter='\t')
df_compound = pd.read_csv('cleared_data/dcid_fingerprint.tsv', delimiter='\t')

df = df_ic50.merge(df_uniprot, on='DeepAffinity Protein ID', how='left')
df = df.merge(df_compound, on='DeepAffinity Compound ID', how='left')

df['label'] = df['pIC50_[M]'] > 8
class_weights = torch.tensor([1 / len(df[~df["label"]]), 1 / len(df[df["label"]])],dtype=torch.float)
print(f'Positive pairs: {len(df[df["label"]])}  |  Negative pairs {len(df[~df["label"]])}')

X_train, X_test, y_train, y_test = train_test_split(df[['Domain Features', 'Fingerprint Feature']], df['label'], test_size=0.2, random_state=42)


# In[37]:
def str_to_series(x):
    return np.array(list(x)).astype('float32')


class MyDataset(Dataset):
  def __init__(self, x, y):
    self.x_1 = torch.tensor(np.stack(x['Fingerprint Feature'].map(str_to_series)))
    self.x_2 = torch.tensor(np.stack(x['Domain Features'].map(str_to_series)))
    self.y = torch.tensor(y.values, dtype=torch.long)

  def __len__(self):
    return len(self.y)

  def __getitem__(self,idx):
    return (self.x_1[idx], self.x_2[idx]), self.y[idx]


# In[38]:


train_ds = MyDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_ds = MyDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


# In[39]:


classes = 2
d_model = 64
s_len_1 = 881
s_len_2 = 16712
n = 3
heads = 2
dropout = 0.1
device = 'cuda'


# In[40]:


model = Model(classes, d_model, s_len_1, s_len_2, n, heads, dropout, device)
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


# In[41]:


losses = pd.DataFrame()
epochs = 10
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
    model.eval()
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(*data)
        loss = criterion(target, labels)
        valid_loss += loss.item()

    losses = pd.concat([losses, pd.DataFrame({'Train_Loss': train_loss / len(train_loader), 'Test_Loss': valid_loss / len(test_loader)}, index=[e])])

    print(f'E {e + 1}  |  Train Loss: {train_loss / len(train_loader) :.3f}  |  Test Loss: {valid_loss / len(test_loader) :.3f}')
    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
        curr_patience = patience
    else:
        curr_patience -= 1
    if curr_patience == 0:
        break


# In[42]:


losses.plot()
plt.show()


# In[43]:

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

y, y_ = [], []

for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        y_.extend(model(*data).detach().cpu().numpy()[:, 1] > 0.5)
        y.extend(labels.detach().cpu().numpy())
print(f'Accuracy: {accuracy_score(y, y_)}  |  Precision {precision_score(y, y_)}  |  Recall {recall_score(y, y_)}')


# In[43]:




