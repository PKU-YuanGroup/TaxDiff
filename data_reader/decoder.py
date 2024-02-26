from __future__ import division, print_function
import glob
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn.utils.rnn import pad_sequence

class Alphabet:
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        x = self.encoding[x]
        return x

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()


class Uniprot21(Alphabet):
    def __init__(self):
        chars = alphabet = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(chars, encoding=encoding)

class decoder_set:
    def __iter__(self):
        return iter(self.x)
    def __init__(self, path='generate_data/protein/generate1.txt', alphabet=Uniprot21()):
        print('# loading array:', path, file=sys.stderr)

        sequences = self.load(path, alphabet)
        # seq_data = []
        # for i in range(len(sequences)):
        #     x =alphabet.decode(sequences[i].astype('uint8'))
        #     seq_data.append(x.decode())
        # self.x = seq_data
        self.x = [[alphabet.decode(seq.astype('uint8')).decode() for seq in sublist] for sublist in sequences]
        print('# decode', len(self.x), 'sequences', file=sys.stderr)
        
    def load(self, path, alphabet):
        data = np.loadtxt(path)
        data = data - np.ones(data.shape)
        # sequence = []
        # star = 0
        # for i in range(data.shape[0]):
        #     where_zero = np.where(data[i] == -1)
        #     for j in range(np.size(where_zero)):
        #         end = where_zero[0][j]
        #         # sequence length fillter
        #         if end - star > 1:
        #             sequence[i].append(data[i][star:end])
        #         star = where_zero[0][j] + 1

        sequence = [[] for _ in range(data.shape[0])]
        for i in range(data.shape[0]):
            star = 0
            where_zero = np.where(data[i] == -1)[0]
            for j in range(len(where_zero)):
                end = where_zero[j]
                # sequence length filter
                if end - star > 1:
                    sequence[i].append(data[i][star:end])
                star = end + 1
            # Check for a sequence after the last -1
            if data.shape[1] - star > 1:
                sequence[i].append(data[i][star:])
            
        return sequence

def len_select(data):
    min_length = 10
    select_seq=[]
    for sublist in data:
        filtered_sublist = [seq for seq in sublist if len(seq) > min_length]
        if filtered_sublist:
            select_seq.append(filtered_sublist)
    return select_seq

    # for i in range(len(data)):
    #     if len(data[i])>10:
    #         select_seq.append(data[i])
    # return select_seq

def random_select(data):
    select_seq=[]
    for i in enumerate(data):
        random.seed(0)
        num = random.randint(0,len(i[1])-1)
        select_seq.append(i[1][num])
    return select_seq

def decode_protein(or_file_path,select_method,gene_num,select_inner):

    file_names = [or_file_path]
    for file_path in file_names:
        data_path = file_path
        data = decoder_set(data_path)
        data = len_select(data.x)
        sum_len = 0
        print("total sequences:",len(data))
        
        # data = random_select(len_data)

        # for i in range(len(data)):
        #     print(i,len(data[i]),data[i])
        #     sum_len += len(data[i])
        # print(sum_len/len(data))

        output_file = file_path.replace("raw_data/", "decode_data/decode_")
        
        if select_method == True:
            with open(output_file, 'w') as f:
                num = 0 
                for i, sublist in enumerate(data):
                    for j in range(len(sublist)):
                        if num == gene_num:
                            break
                        else:
                            f.write(f">sequence_{num}_{i}_{j}\n"+sublist[j]+f"\n")
                            num +=1
            print(f"Data save to {output_file}")
        else:
            if select_inner == True:
                random.seed(0)
            with open(output_file, 'w') as f:
                for i, sublist in enumerate(data):
                    j = random.randint(0, len(sublist)-1)
                    f.write(f">sequence_{i}_{j}\n"+sublist[j]+f"\n")
            print(f"Data save to {output_file}")

if __name__ == '__main__':
    file_names = [
    #  '/remote-home/lzy/DiT/results_tid/Ture-taxid-200-256/001-DiT-S-2/raw_data/tid11_epoch1_100.txt'
    '/remote-home/lzy/DiT/results_tid/Ture-taxid-200-256/001-DiT-S-2/raw_data/tid23428_epoch1_100.txt',]
    decode_protein('/remote-home/lzy/DiT/results_tid/Ture-taxid-200-256/001-DiT-S-2-p2/raw_data/random_epoch1_500.txt',False,1000)
    





