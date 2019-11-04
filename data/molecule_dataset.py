import numpy as np
import torch
from torch.utils.data import Dataset

from utilities import ops, settings

def _get_seq_lens(x):
    if isinstance(x, list):
        return [int((xs == 0).max(0)[1]) if 0 in xs else len(xs) for xs in x]
    else:
        return (x == 0).max(1)[1]


def _pad_seqs(x, max_len):
    x_pad = torch.LongTensor(len(x), max_len).zero_()
    for n, (seq, seq_len) in enumerate(zip(x, _get_seq_lens(x))):
        x_pad[n, :seq_len] = seq[:seq_len]
    return x_pad


class ZincDataset(Dataset):
    def __init__(self, data_file, fname='train'):
        super().__init__()
        self.name = data_file.split('/')[-1].split('.')[0]
        idxs = np.load(data_file)[fname]
        if len(idxs) == 0:
            raise ValueError('No data in {}:{}'.format(data_file, fname))
        idxs = [torch.from_numpy(i) for i in idxs]
        self.idxs = _pad_seqs(idxs, settings.max_len)

    def __getitem__(self, item):
        return self.idxs[item]

    def __len__(self):
        return len(self.idxs)


class PerturbedZincDataset(ZincDataset):
    def __init__(self, data_file, alphabet, p_perturbed=0.05, fname='train'):
        super().__init__(data_file, fname)
        self.alphabet = alphabet
        self.p_perturbed = p_perturbed
        print('PerturbedZincDataset len:', len(self.idxs))

    def __getitem__(self, item):
        x = self.alphabet.generate_perturbed(super().__getitem__(item), self.p_perturbed)
        return x, self.alphabet.validate(x)


class PrefixZincDataset(ZincDataset):
    def __init__(self, data_file, alphabet, fname='train'):
        super().__init__(data_file, fname)
        self.alphabet = alphabet
        print('PrefixZincDataset len:', len(self.idxs))

    def __getitem__(self, item):
        x = self.alphabet.generate_perturbed(super().__getitem__(item), self.p_perturbed)
        return x, self.alphabet.validate(x)
