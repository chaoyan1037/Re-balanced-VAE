import argparse
import re
import numpy as np
import pickle
from rdkit import Chem

import sys
sys.path.append("..")

from grammar_model.alphabets import MoleculeAlphabet
a = MoleculeAlphabet()
data_file = 'zinc.smi'


with open(data_file, 'r') as fhandle:
    mol_strs = fhandle.read().strip().split('\n')

rejected = 0
print('mol_strs len:', len(mol_strs), mol_strs[:10])
# for mol_str in mol_strs:
#     mol = Chem.MolFromSmiles(mol_str)
#     if mol is None:
#         rejected += 1
# print('rejected', rejected)

def smi_tokenizer(regex, smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
mol_strs = [smi_tokenizer(regex, sm) for sm in mol_strs]
print('mol_strs:', mol_strs[:10])

# with open('zinc.train.txt', 'w') as f:
#     for s in mol_strs[:-5000]:
#         f.write(s + '\n')

# with open('zinc.test.txt', 'w') as f:
#     for s in mol_strs[-5000:]:
#         f.write(s + '\n')

tokens = set()
max_len = 0
for mol_str in mol_strs:
    mol_str = mol_str.split(' ')
    if len(mol_str) > 72:
        print(mol_str)
    max_len = max(max_len, len(mol_str))
    tokens.update(set(mol_str))
print('max_len:', max_len)
print('tokens:', len(tokens), tokens)

with open('vocab.txt', 'w') as f:
    for v in tokens:
        f.write(v + '\n')

# tokens = [' ', 'UNKNOWN'] + list(tokens)
# print('tokens:', len(tokens), tokens)

# idx_to_token = dict()
# token_to_idx = dict()

# for idx, token in enumerate(tokens):
#     idx_to_token[idx] = token
#     token_to_idx[token] = idx
# print(idx_to_token, token_to_idx)

# mol_tokens = []
# for mol_str in mol_strs:
#     mol_str = mol_str.split(' ')
#     mol_token = [token_to_idx[t] for t in mol_str]
#     mol_token = mol_token + [token_to_idx[' ']] * (max_len - len(mol_token))
#     mol_tokens.append(mol_token)

# data = np.array(mol_tokens, dtype=np.uint8)

# save_data = {'train': data[:-5000], 'test': data[-5000:], 'idx_to_token': idx_to_token, 'token_to_idx': token_to_idx}

# with open("zinc_tokenized", "wb") as f:
#     pickle.dump(save_data, f)
