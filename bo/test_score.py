import numpy as np
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer
import networkx as nx
import os


smiles = ['O=C(c1ccccc1Sc1ccc(Cl)cc1)c1ccc(-c2ccccc2Cl)cc1',
          'O=C(Nc1cccc(Sc2nc3ccc(Br)cc3n2-c2ccccc2)c1)c1ccc(Cl)cc1',
          'O=C(Nc1cccc(Sc2nc3ccc(Br)cc3n2-c2ccccc2)c1)c1ccc(Cl)cc1',
          'O=C(Nc1cccc(Sc2ccc(-c3ccccc3)cn2)c1)c1ccc(Cl)cc1Cl',
          'O=C(Nc1cccc(Oc2ccc(Br)cc2)c1)c1ccc2cc(Cl)ccc2c1',
          'C(Nc1cccc(Oc2ccc(-c3cccs3)cc2)c1)c1ccc(Cl)cc1']


cwd = os.getcwd()
print('cwd:', cwd)
logP_values = np.loadtxt(os.path.join(cwd, 'logP_values.txt'))
logP_values_mean, logP_values_var = np.mean(logP_values), np.std(logP_values)
SA_scores = np.loadtxt(os.path.join(cwd, 'SA_scores.txt'))
SA_scores_mean, SA_scores_var = np.mean(SA_scores), np.std(SA_scores)
cycle_scores = np.loadtxt(os.path.join(cwd, 'cycle_scores.txt'))
cycle_scores_mean, cycle_scores_var = np.mean(cycle_scores), np.std(cycle_scores)


logP_values = []
SA_scores = []
cycle_scores = []
for smi in smiles:
    mol = MolFromSmiles(smi)
    if mol is None:
        print('Mol is None:', smi)
        continue
    logP_values.append(Descriptors.MolLogP(mol))
    SA_scores.append(-sascorer.calculateScore(mol))

    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

print('logP_values:', logP_values)
print('SA_scores:', SA_scores)
print('cycle_scores:', cycle_scores)

SA_scores_normalized = (SA_scores - SA_scores_mean) / SA_scores_var
logP_values_normalized = (logP_values - logP_values_mean) / logP_values_var
cycle_scores_normalized = (cycle_scores - cycle_scores_mean) / cycle_scores_var

scores = []
for i in range(len(logP_values)):
    scores.append(SA_scores_normalized[i] + logP_values_normalized[i] + cycle_scores_normalized[i])

print(smiles)
print(scores)
