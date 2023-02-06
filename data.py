import json
import os
import shutil
import time

import gdown
import pandas as pd
from datasets import Dataset

data_url = 'https://drive.google.com/uc?id=1nJIaGgxWMimo1x2868f3l9PfJ79ktBIE'
data_output = 'data.zip'
data_files = ['IC50_protein_compound_pair.tsv', 'dpid_seq.tsv', 'dcid_smi.tsv', 'embeddings.json']

dataset_dir = 'dataset'


def download_if_needed():
    for file in data_files:
        if not os.path.isfile(file):
            gdown.download(data_url, data_output, quiet=False)
            shutil.unpack_archive(data_output, '.')
            break

# Not used but left as an example of how embeddings.json was generated
# this takes a pretty long time because the model is massive
# use environmnet from docker, kubenoz/ipz:latest, to get an environment configured that works
# with jax_unirep, it might not work with the rest of the code so I suggest copying i to a separate script
# and running that instead
# The implementation also holds onto reserved GPU memory if cancelled prematurelly
# If that happens, wait a few seconds before restarting or free the gpu memory e.g. by allocating it
# to a different application
"""
from jax_unirep import get_reps

def process_data():
    df = pd.read_csv('cleared_data/dpid_seq.tsv', delimiter='\t')
    ptime_total = time.process_time()
    with open('cleared_data/embeddings.json', 'w') as resfile:
        i = 0
        while i < len(df.index):
            ptime = time.process_time()
            df_temp = df[i:min(i+100, len(df.index))]
            res = df_temp['Sequence'].to_numpy()
            res, _, _ = get_reps(res)
            res = res.tolist()
            labels = df_temp['DeepAffinity Protein ID'].tolist()
            res_lst = []
            for label, res_i in zip(labels, res):
                res_dict = {label: res_i}
                json.dump(res_dict, resfile)
                resfile.write('\n')
            ptime = time.process_time() - ptime
            print(f'index {i} to {min(i+100, len(df.index))} done.\nTime taken was {ptime//60} minutes and {ptime%60} seconds.')
            i += 100
    ptime_total = time.process_time() - ptime_total
    print(f'All done! Total time taken was {ptime_total//60} minutes and {ptime_total%60} seconds.')
"""

def pandas_data():
    df_ic50 = pd.read_csv('cleared_data/IC50_protein_compound_pair.tsv', delimiter='\t')
    df_uniprot = pd.read_json('cleared_data/embeddings.json', lines=True)
    df_compound = pd.read_csv('cleared_data/dcid_smi.tsv', delimiter='\t')
    df = df_ic50.merge(df_uniprot, on='DeepAffinity Protein ID', how='inner')
    df = df.merge(df_compound, on='DeepAffinity Compound ID', how='inner')
    df['label'] = df['pIC50_[M]'] >= 5
    df['label'] = df['label'].astype(float)

    #df['embeddings'] = df['Sequence'].map(get_reps) if you have a lot of ram, >64gb, you can try
    # using this instead of the method above
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(dataset_dir)

if __name__ == '__main__':
    download_if_needed()
    pandas_data()
