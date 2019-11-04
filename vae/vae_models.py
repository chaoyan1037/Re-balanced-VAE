import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import partialsmiles as ps

from rdkit import Chem
from utilities import ops

class Vae(nn.Module):
    def __init__(self, vocab, vocab_size, embedding_size, dropout, padding_idx,
            sos_idx, unk_idx, max_len, n_layers, hidden_size,
            bidirectional=False, latent_size=128, partialsmiles=False):
        super(Vae, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.padding_idx = padding_idx  # eos padding 
        self.sos_idx = sos_idx
        self.unk_idx = unk_idx
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.latent_size = latent_size
        self.dropout = dropout
        self.partialsmiles = partialsmiles

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)
        
        self.encoder = Encoder(input_size=embedding_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               bidirectional=bidirectional,
                               latent_size=latent_size)

        dec_layers = (2 if bidirectional else 1) * n_layers
        self.decoder = Decoder(input_size=embedding_size,
                               hidden_size=hidden_size,
                               n_layers=dec_layers,
                               dropout=dropout,
                               latent_size=latent_size,
                               vocab_size=vocab_size,
                               max_len=max_len,
                               vocab=vocab,
                               sos_idx=sos_idx,
                               padding_idx=padding_idx)

        self.encoder_params = list(self.embedding.parameters()) + list(self.encoder.parameters())
        self.decoder_params = self.decoder.parameters()

    def forward(self, input_sequence, epsilon_std=1.0):
        # input sequence are like: <sos> SMILES tokens <padding>
        # for input, remove the last token
        input_sequence = input_sequence[:, :-1]
        input_embedding, mean, logv, z = self.encoder_sample(input_sequence, epsilon_std)

        # decoder output projection on vacab
        outputs = self.decoder(input_embedding, z)
        return outputs, mean, logv, z

    def encoder_sample(self, input_sequence, epsilon_std=1.0):
        batch_size = input_sequence.size(0)
        # move data onto GPU
        input_sequence = input_sequence.cuda()
        
        # embedding input sequences
        input_embedding = self.embedding(input_sequence)
        
        # encoder process
        mean, logv = self.encoder(input_embedding)
        
        # sample prior
        z = ops.sample(mean, logv, (batch_size, self.latent_size), epsilon_std).cuda()
        return input_embedding, mean, logv, z

    def inference(self, latent, max_len=None):
        # decoder inference
        outputs = self.decoder.inference(latent, self.embedding, max_len=max_len, partialsmiles=self.partialsmiles)
        return outputs

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """
        # [x_batch, nz]
        _, mu, logvar, z_samples = self.encoder_sample(x)
        x_batch, nz = mu.size()
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()
        var = logvar.exp()
        # (z_batch, 1, nz)
        z_samples = z_samples.unsqueeze(1)
        # (1, x_batch, nz)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)
        # (z_batch, x_batch, nz)
        dev = z_samples - mu
        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
        # log q(z): aggregate posterior
        log_qz = ops.log_sum_exp(log_density, dim=1) - math.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()


class Encoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, n_layers=1,
                 bidirectional=False, latent_size=128):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.latent_size = latent_size
        self.hidden_factor = (2 if bidirectional else 1) * n_layers

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True)

        self.mean_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.logvar_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        ops.init_params(self)


    def forward(self, input_embedding):
        """
        :param input_embedding: [batch_size, seq_len, embed_size] tensor
        :return: latent vector mean and log var [batchsize, latentsize] 
        """
        _, hidden = self.rnn(input_embedding)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.size(0), -1)

        # reparameterization
        mean = self.mean_lin(hidden)
        logv = -torch.abs(self.logvar_lin(hidden))
        
        return mean, logv


class Decoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, n_layers=1,
                 dropout=0.5, latent_size=128,
                 vocab_size=64, max_len=75, vocab=None, sos_idx=2, padding_idx=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.vocab = vocab
        self.sos_idx = sos_idx
        self.padding_idx = padding_idx
        self.hidden_factor = n_layers

        self.embedding_dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True)

        self.latent2hidden = torch.nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = torch.nn.Linear(hidden_size, vocab_size)
        self.outputs_dropout = nn.Dropout(p=dropout)

        ops.init_params(self)

    def forward(self, input_embedding, latent):
        hidden = self.latent2hidden(latent)
        
        hidden = hidden.view(-1, self.hidden_factor, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)

        input_embedding = self.embedding_dropout(input_embedding)
        outputs, _ = self.rnn(input_embedding, hidden)

        # process outputs
        b, seq_len, hsize = outputs.size()
        outputs = outputs.contiguous().view(-1, hsize)
        
        outputs = self.outputs_dropout(outputs)
        outputs = self.outputs2vocab(outputs)
        
        return outputs.view(b, seq_len, self.vocab_size)

    def inference_guided(self, latent, embedding, max_len=None):
        if max_len is None:
            max_len = self.max_len
        assert latent.size(1) == self.latent_size, 'latent size error!'
        batch_size = latent.size(0)
        hidden = self.latent2hidden(latent)

        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)

        input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).unsqueeze(1).long()
        index_pred = torch.LongTensor()
        smiles_pred = ["" for _ in range(batch_size)]
        smiles_state = [0 for _ in range(batch_size)]  # -1: failed, 1; succeed, 0: todo
        for t in range(max_len):
            input_sequence = input_sequence.cuda()
            input_embedding = embedding(input_sequence)

            # decoder rnn run once
            output, hidden = self.rnn(input_embedding, hidden)
            output = self.outputs_dropout(output)
            logits = self.outputs2vocab(output).cpu()
            
            # prepare next input
            input_sequence = torch.argmax(logits, dim=-1)
            _, index = torch.sort(logits, dim=-1, descending=True)
            for i in range(batch_size):
                # check if the smiles finished
                if smiles_state[i] != 0:
                    continue
                # check for all tokens in descending possibility
                flag = False
                for j in range(logits.size(-1)):
                    idx_cur = index[i][0][j]
                    if idx_cur == self.padding_idx:
                        if Chem.MolFromSmiles(smiles_pred[i]) is not None:
                            input_sequence[i] = idx_cur
                            smiles_state[i] = 1
                            break
                    elif idx_cur > 2:
                        smi = smiles_pred[i] + self.vocab.itos[idx_cur]
                        try:
                            ps.ParseSmiles(smi, partial=True)
                            input_sequence[i] = idx_cur
                            smiles_pred[i] = smi
                            flag = True
                            break
                        except ps.Error as e:
                            continue
                # failed for all tokens
                if not flag:
                    smiles_state[i] = -1
            index_pred = torch.cat((index_pred, input_sequence), dim=1)
            if 0 not in smiles_state:
                break
        return smiles_pred, index_pred

    def inference_direct(self, latent, embedding, max_len=None):
        if max_len is None:
            max_len = self.max_len
        assert latent.size(1) == self.latent_size, 'latent size error!'
        batch_size = latent.size(0)
        hidden = self.latent2hidden(latent)

        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)

        input_sequence = torch.Tensor(batch_size).fill_(self.padding_idx).unsqueeze(1).long()
        logits_t = torch.FloatTensor()
        for t in range(max_len):
            input_sequence = input_sequence.cuda()
            input_embedding = embedding(input_sequence)
            # decoder rnn run once
            output, hidden = self.rnn(input_embedding, hidden)
            output = self.outputs_dropout(output)
            logits = self.outputs2vocab(output).cpu()
            logits_t = torch.cat((logits_t, logits), dim=1)
            # prepare next input
            input_sequence = torch.argmax(logits, dim=-1)

        index_pred = torch.argmax(logits_t, dim=-1)
        smiles_pred = []
        for i in range(batch_size):
            smi = [self.vocab.itos[p] for p in index_pred[i]]
            smi = ''.join(smi).split()[0]
            smiles_pred.append(smi)
        return smiles_pred, index_pred


    def inference(self, latent, embedding, max_len=None, partialsmiles=False):
        if partialsmiles:
            return self.inference_guided(latent, embedding, max_len)
        else:
            return self.inference_direct(latent, embedding, max_len)
        

