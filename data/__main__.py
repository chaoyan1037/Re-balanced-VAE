
import argparse
from rdkit import Chem
import sys
sys.path.append(".")

import numpy as np

from grammar_model.alphabets import MoleculeAlphabet

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
args = parser.parse_args()

a = MoleculeAlphabet()

with open(args.data, 'r') as fhandle:
    mol_strs = fhandle.read().strip().split('\n')

print("Preprocessing, this takes a couple minutes")
clean_strs, clean_idxs = [], []

rejected = 0
for mol_str in mol_strs:
    if not a._validate(mol_str):
        rejected += 1
        continue
    try:
        idx = a.expr2idx(mol_str)
        clean_strs.append(mol_str)
        clean_idxs.append(idx)
    except ValueError:
        rejected += 1
        continue

max_len = np.max([len(idx) for idx in clean_idxs])
print('max_len:', max_len)
max_len = 75
idxs_padding = []
for idx in clean_idxs:
    idxs_padding.append(idx + [a.alphabet.index(' ')] * (max_len - len(idx)))

data = np.array(idxs_padding, dtype=np.uint8)

if "zinc" in args.data:
    print(len(data))
    np.savez_compressed(args.data.split('.')[0], train=data[:-5000], test=data[-5000:])
else:
    np.savez_compressed(args.data.split('.')[0], test=data)

print("File {} processed, {} rejected.".format(args.data, rejected))

