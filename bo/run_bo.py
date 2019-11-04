import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os.path
import sys

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
import torch.nn.functional as F

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

# Load model
if not '/home/chaoyan/Documents/DL/beta_vae_mol' in sys.path:
  sys.path += ['/home/chaoyan/Documents/DL/beta_vae_mol']
print('sys.path:\n', sys.path)

from data.molecule_iterator import SmileBucketIterator
from vae import vae_models
from vae.vae_trainer import VAETrainer, VAEArgParser
parser = VAEArgParser()
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--random_seed', type=int, default=1)

args = parser.parse_args()
print(args)


random_seed = args.random_seed
# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minued!)
X = np.loadtxt('latent_features.txt')
y = -np.loadtxt('targets.txt')
y = y.reshape((-1, 1))

# X = X[:1000]
# y = y[:1000]

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]
print('X_train', type(X_train), X_train.shape)
print('y_train', type(y_train), y_train.shape)

SA_scores = np.loadtxt('SA_scores.txt')
logP_values = np.loadtxt('logP_values.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')

SA_scores_mean, SA_scores_var = np.mean(SA_scores), np.std(SA_scores)
logP_values_mean, logP_values_var = np.mean(logP_values), np.std(logP_values)
cycle_scores_mean, cycle_scores_var = np.mean(cycle_scores), np.std(cycle_scores)

SA_scores_normalized = (np.array(SA_scores) - SA_scores_mean) / SA_scores_var
logP_values_normalized = (np.array(logP_values) - logP_values_mean) / logP_values_var
cycle_scores_normalized = (np.array(cycle_scores) - cycle_scores_mean) / cycle_scores_var

print(len(logP_values), len(SA_scores), len(cycle_scores))



smi_file = 'zinc_smile.txt'
vocab_file = '../data/vacab'

smi_iterator = SmileBucketIterator(smi_file, vocab_file, args.batch_size)
vocab_size = smi_iterator.vocab_size
padding_idx = smi_iterator.padding_idx
sos_idx = smi_iterator.sos_idx
eos_idx = smi_iterator.eos_idx
unk_idx = smi_iterator.unk_idx
print('vocab_size:', vocab_size)
print('padding_idx sos_idx eos_idx unk_idx:', padding_idx, sos_idx, eos_idx, unk_idx)
vocab = smi_iterator.get_vocab()
# define Vae model
vae = vae_models.Vae(vocab, vocab_size,  args.embedding_size, args.dropout,
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


iteration = 0
while iteration < 5:
    # We fit the GP
    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 10, learning_rate = 0.001)
   
    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    
    mol_vec = torch.from_numpy(next_inputs).float().cuda()
    smiles_pred, _ = vae.inference(mol_vec)
    for i, smi in enumerate(smiles_pred):
        if not smi.startswith("JUNK") and MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)
            new_features.append(next_inputs[i])

    print(len(valid_smiles), "molecules are found")
    valid_smiles = valid_smiles[:50]
    new_features = next_inputs[:min(50, len(valid_smiles))]
    new_features = np.vstack(new_features)
    save_object(valid_smiles, args.save_dir + "/valid_smiles{}.dat".format(iteration))
    print('valid_smiles:', len(valid_smiles))
    print('new_features:', new_features.shape)

    import sascorer
    import networkx as nx
    from rdkit.Chem import rdmolops

    scores = []
    for i in range(len(valid_smiles)):
        mol = MolFromSmiles(valid_smiles[i])
        current_log_P_value = Descriptors.MolLogP(mol)
        current_SA_score = -sascorer.calculateScore(mol)
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        current_cycle_score = -cycle_length
        current_SA_score_normalized = (current_SA_score - SA_scores_mean) / SA_scores_var
        current_log_P_value_normalized = (current_log_P_value - logP_values_mean) / logP_values_var
        current_cycle_score_normalized = (current_cycle_score - cycle_scores_mean) / cycle_scores_var

        score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
        scores.append(-score) #target is always minused

    print(valid_smiles)
    print(scores)
    print("best score:", min(scores))
    print(sorted(scores))

    save_object(scores, args.save_dir + "/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
