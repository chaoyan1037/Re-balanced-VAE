import numpy
import pickle
import re
import sys
import torchtext
from torchtext.data import Example, Field, Dataset
from torchtext.data import Iterator, BucketIterator

pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
def smi_tokenizer(smi, regex=re.compile(pattern)):
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

smi_field = Field(sequential=True, init_token='<sos>', eos_token=' ', pad_token=' ',
                  include_lengths=True, batch_first=True, tokenize=smi_tokenizer)

with open('zinc.smi', 'r') as f:
    mol_strs = f.read().strip().split('\n')
mol_strs = [smi_field.preprocess(mol) for mol in mol_strs]


smi_field.build_vocab(mol_strs)
print(len(smi_field.vocab.itos))
print(smi_field.vocab.itos)
print(smi_field.vocab.stoi)
input()

# with open('vocab', 'wb') as f:
#     pickle.dump(smi_field.vocab, f)
# vocab = pickle.load(open('vocab', 'rb'))
# tokens = vocab.freqs.keys()
# print(len(tokens))
# print(tokens)
# print(vocab.itos)

# print('__eq__', vocab == smi_field.vocab)


smi_examples = []
for mol in mol_strs:
    ex = Example.fromlist([mol], [('smile', smi_field)])
    smi_examples.append(ex)

smi_data = Dataset(smi_examples, fields=[('smile', smi_field)])

print(smi_data.__dict__.keys())
print(smi_data[0].smile)

train_iter = BucketIterator(smi_data, batch_size=256, train=True, 
                            sort_within_batch=True, repeat=False,
                            sort_key=lambda x: len(x.smile))

numpy.set_printoptions(threshold=sys.maxsize)

for _ in range(2):
    for idx, batch in enumerate(train_iter):
        print(idx, batch.smile[0].shape, batch.smile[1].shape)
        print(batch.smile[0])
