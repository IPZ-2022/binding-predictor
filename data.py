import json
import os

import datasets
import gdown
import shutil
import pandas as pd
import psutil
import torch
from datasets import DatasetDict, load_from_disk, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from jax_unirep import get_reps
import time
import multiprocessing
from transformers import AutoTokenizer, AutoModel
import pyarrow as pa
from pyarrow import csv
from pyarrow import json as ajson
import encodings
import jsonlines
data_url = 'https://drive.google.com/uc?id=1nJIaGgxWMimo1x2868f3l9PfJ79ktBIE'
data_output = 'data.zip'
data_files = ['cleared_data/IC50_protein_compound_pair.tsv', 'cleared_data/dpid_seq.tsv', 'cleared_data/dcid_smi.tsv', 'embeddings.json']

dataset_dir = 'dataset'


def download_if_needed():
    gdown.download(data_url, data_output, quiet=False)
    shutil.unpack_archive(data_output, 'cleared_data')

def get_data():
    if not os.path.isdir(dataset_dir):
        for filepath in data_files:
            if not os.path.isfile(filepath):
                download_if_needed()
                break
    pandas_data()




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


def correct_data():
    schema = pa.schema([('DeepAffinity Protein ID', pa.string()), ('embedding', pa.list_(pa.float32(), 1900))])
    json_opts = ajson.ParseOptions(explicit_schema=schema)
    t_emb = ajson.read_json('cleared_data/embeddings.json')
    emb_col = t_emb['embedding']
    num_vals = emb_col.to_numpy()
    num_vals = np.concatenate(num_vals)
    typ = pa.list_(pa.float64(), 1900)
    arr = pa.FixedSizeListArray.from_arrays(num_vals, list_size=1900)
    t_emb = t_emb.drop(['embedding'])
    t_emb = t_emb.append_column('embedding', arr)


    csv_opts = csv.ParseOptions(delimiter='\t')
    t_conn = csv.read_csv('cleared_data/IC50_protein_compound_pair.tsv', parse_options=csv_opts)
    t_smi = csv.read_csv('cleared_data/dcid_smi.tsv', parse_options=csv_opts)
    t_join = t_smi.join(right_table=t_conn, keys='DeepAffinity Compound ID', join_type='inner')
    t_join1 = t_join.join(right_table=t_emb, keys='DeepAffinity Protein ID', join_type='inner')

    print(psutil.virtual_memory())
    print('test done')


def pandas_data():
    df_ic50 = pd.read_csv('cleared_data/IC50_protein_compound_pair.tsv', delimiter='\t')
    df_uniprot = pd.read_json('cleared_data/embeddings.json', lines=True)
    df_compound = pd.read_csv('cleared_data/dcid_smi.tsv', delimiter='\t')
    df = df_ic50.merge(df_uniprot, on='DeepAffinity Protein ID', how='inner')
    df = df.merge(df_compound, on='DeepAffinity Compound ID', how='inner')
    df['label'] = df['pIC50_[M]'] >= 5
    df['label'] = df['label'].astype(float)

    #df['embeddings'] = df['Sequence'].map(get_reps)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(dataset_dir)


def map_reps(x):
    temp, _, _ = get_reps(x['Sequence'])
    temp = [t for t in temp]
    x['embedding'] = temp
    return x

if __name__ == '__main__':
    pandas_data()
