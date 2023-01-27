import os

import datasets
import gdown
import shutil
import pandas as pd
from datasets import DatasetDict, load_from_disk, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from jax_unirep import get_reps
import time
import multiprocessing

data_url = 'https://drive.google.com/uc?id=1_msEbSh_YZr0NLSR_DJ_xWE9FlqBlMV9'
data_output = 'data.tgz'
data_files = ['cleared_data/IC50_protein_compound_pair.tsv', 'cleared_data/dpid_seq.tsv', 'cleared_data/dcid_smi.tsv']

dataset_dir = 'dataset'


def get_data():
    if not os.path.isfile(data_output):
        gdown.download(data_url, data_output, quiet=False)

    if not os.path.isdir('cleared_data'):
        shutil.unpack_archive(data_output, 'cleared_data')
    else:
        for filepath in data_files:
            if not os.path.isfile(filepath):
                shutil.unpack_archive(data_output, 'cleared_data')
                break
    shutil.rmtree('cleared_data/SDF', ignore_errors=True)

    if not os.path.isdir(dataset_dir):
        return process_data()

    return load_from_disk(dataset_dir)


def process_data():
    if not os.path.isdir('temp_dataset'):
        df_ic50 = pd.read_csv('cleared_data/IC50_protein_compound_pair.tsv', delimiter='\t')
        df_uniprot = datasets.load_from_disk('cleared_data/dpid_seq.tsv', delimiter='\t')

        df_compound = pd.read_csv('cleared_data/dcid_smi.tsv', delimiter='\t')
        df = df_ic50.merge(df_uniprot, on='DeepAffinity Protein ID', how='left')
        df = df.merge(df_compound, on='DeepAffinity Compound ID', how='left')
        df['label'] = df['pIC50_[M]'] >= 5
        df['label'] = df['label'].astype(float)
        df = df[['Sequence', 'Canonical SMILE', 'label']]

        #df['embeddings'] = df['Sequence'].map(get_reps)

        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk('temp_dataset')


    dataset = load_from_disk('temp_dataset')
    ptime = time.process_time()
    print(f'starting embedding dataset of size {len(dataset)}, this might take a while')
    dataset = dataset.map(map_reps, remove_columns=['Sequence'])
    ptime = time.process_time() - ptime
    print(f'Embedding finished in {ptime//60} minutes and {round(ptime%60)} seconds')
    dataset.save_to_disk(dataset_dir)
    shutil.rmtree('temp_dataset', ignore_errors=True)
    return dataset

def map_reps(x):
    temp, _, _ = get_reps(x['Sequence'])
    temp = [t for t in temp]
    x['embedding'] = temp
    return x

if __name__ == '__main__':
    get_data()
