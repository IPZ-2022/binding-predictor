import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import gdown
import os

from model.model import Model

classes = 2
d_model = 64
s_len_1 = 768
s_len_2 = 1900
n = 3
heads = 2
dropout = 0.1
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

req_files = ['dcid_smi.tsv', 'nsp13-encoded.npy', 'best_model.pth']
url_prefix = 'https://drive.google.com/uc?id='
urls = ['1eJb1_uiAS605fyIjJK5n7-zFCB-m_sFi', '1UjYU8qbgrD7qPdAZZB8CLt-rdUWPxg9d', '1TzpA5MQqwQQP0Dj8Bm7mnK1Os4G-6f2P']

for filepath, url in zip(req_files, urls):
    if not os.path.isfile(filepath):
        gdown.download(url_prefix+url, filepath)

model = Model(classes, d_model, s_len_1, s_len_2, n, heads, dropout, device)
state_fixed = OrderedDict()
state = torch.load('best_model.pth')
for k, v in state.items():
    state_fixed[k[7:]] = v
model.load_state_dict(state_fixed)
model.to(device)
model.eval()


df_compound = pd.read_csv('dcid_smi.tsv', delimiter='\t')

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

dataset = Dataset.from_pandas(df_compound)
dataset = dataset.map(lambda e: tokenizer(e['Canonical SMILE'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'DeepAffinity Compound ID', 'Canonical SMILE'])


nsp13_encoded = np.load('nsp13-encoded.npy')

nsp13_encoded = torch.tensor(nsp13_encoded)
nsp13_encoded = nsp13_encoded.repeat(batch_size, 1)

data_loader = DataLoader(dataset, batch_size=batch_size)

records = {'ids':[], 'vals':[]}

names = []
vals = []
smiles = []
i = 0
timep = time.process_time()
for x in data_loader:
    x['embedding'] = nsp13_encoded
    out = model(x, device)
    out = torch.nn.functional.softmax(out, dim=1)
    out = out[:,1].tolist()
    for name, smile, val in zip(x['DeepAffinity Compound ID'], x['Canonical SMILE'], out):
        names.append(name)
        smiles.append(smile)
        vals.append(val)
    if i % 100 == 0:
        timep = time.process_time()-timep
        print(f'Done {i}, it took {timep}')
        timep = time.process_time()
    i += 1

with open('results.csv', 'w') as resfile:
    resfile.write('DeepAffinity Compound ID,Canonical SMILE,val\n')
    for name, smile, val in zip(names, smiles, vals):
        resfile.write(f'{name},{smile},{val}\n')
