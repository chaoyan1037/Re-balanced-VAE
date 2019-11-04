import numpy as np 
import sascorer
import sys
import torch
import torch.nn as nn

from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops


# lg = rdkit.RDLogger.logger()
# lg.setLevel(rdkit.RDLogger.CRITICAL)

smi_file = '../data/zinc.smi'
vocab_file = '../data/vacab'

if not '/home/chaoyan/Documents/DL/beta_vae_mol' in sys.path:
  sys.path += ['/home/chaoyan/Documents/DL/beta_vae_mol']
print ('sys.path:\n', sys.path)

from data.molecule_iterator import SmileBucketIterator
from vae import vae_models
from vae.vae_trainer import VAETrainer, VAEArgParser
args = VAEArgParser().parse_args()
print(args)


smi_iterator = SmileBucketIterator(smi_file, vocab_file, args.batch_size)
    
if args.test_mode:
    smi_iterator.train_smi.examples = smi_iterator.train_smi.examples[:1000]

dataset_bucket_iter = smi_iterator.dataset_bucket_iter()
vocab_size = smi_iterator.vocab_size
padding_idx = smi_iterator.padding_idx
sos_idx = smi_iterator.sos_idx
eos_idx = smi_iterator.eos_idx
unk_idx = smi_iterator.unk_idx
print('vocab_size:', vocab_size)
print('padding_idx sos_idx eos_idx unk_idx:', padding_idx, sos_idx, eos_idx, unk_idx)
vocab = smi_iterator.get_vocab()
# define Vae model
vae = vae_models.Vae(vocab, vocab_size, args.embedding_size, args.dropout,
                     padding_idx, sos_idx, unk_idx,
                     args.max_len, args.n_layers, args.layer_size,
                     bidirectional=args.enc_bidir, latent_size=args.latent_size)
vae = vae.cuda()
if args.restore:
    state = torch.load(args.restore)
    vae.load_state_dict(state['vae'])
else:
    print("error! empty restore")
    input()

smiles = []
latent_points = []
vae.train()
for batch in dataset_bucket_iter:
    x = batch.smile[0]
    _, _, _, z_mean = vae.encoder_sample(x, epsilon_std=1.0)
    
    for i in range(x.size(0)):
        x_k = x[i].cpu().numpy().tolist()
        x_k = [vocab.itos[p] for p in x_k]
        x_k = "".join(x_k[1:]).replace(" ", "")
        smiles.append(x_k)

    latent_points += z_mean.cpu().detach().numpy().tolist()

print(len(latent_points), len(smiles))

# We store the results
latent_points = np.vstack(latent_points)
print('latent_points shape:', latent_points.shape)
import os
cwd = os.getcwd()
np.savetxt(os.path.join(cwd, 'latent_features.txt'), latent_points)

#smiles = smiles[:500]

smiles_rdkit = []
for i in range(len(smiles)):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ]), isomericSmiles=True))

logP_values = []
for i in range(len(smiles)):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))

SA_scores = []
for i in range(len(smiles)):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))

import networkx as nx

cycle_scores = []
for i in range(len(smiles)):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)



targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt(os.path.join(cwd, 'targets.txt'), targets)
np.savetxt(os.path.join(cwd, 'logP_values.txt'), np.array(logP_values))
np.savetxt(os.path.join(cwd, 'SA_scores.txt'), np.array(SA_scores))
np.savetxt(os.path.join(cwd, 'cycle_scores.txt'), np.array(cycle_scores))

with open(os.path.join(cwd, "zinc_smile.txt"), "w") as f:
    for smi in smiles:
        f.write(smi + "\n")
