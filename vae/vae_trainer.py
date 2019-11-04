import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from rdkit import Chem
from base_classes import schedulers
from base_classes.trainer import TrainerBase, TrainerArgParser
from utilities import ops


weight = {'o': 1.4554088613628926, '[C@]': 1.5435154133217646, '(': 1.1336555900413336, '2': 1.2271525729258035, '[NH+]': 1.4465062838107283, 'I': 1.6430956542751018, '[n-]': 1.6867424058204643, 'F': 1.3881706845994657, '[P@@]': 1.7699859535284932, '[O-]': 1.4710446425117816, 'O': 1.2114536486650351, '/': 1.46070186168524, '[S@@]': 1.635572566873708, 's': 1.4396772582455102, 'C': 1.0345098327192341, 'S': 1.4077282065547054, '[P@]': 1.7776429338887791, '[S-]': 1.6717204720988241, '=': 1.2422088634832562, '-': 1.442980341166416, '[S@]': 1.6344459064391046, '[nH+]': 1.5624020603137296, '4': 1.4558197854433321, '[o+]': 1.7963122129799949, '1': 1.1714583077479712, '[n+]': 1.6506253800862147, '[C@@H]': 1.3663339467456452, '7': 1.808367246924444, 'n': 1.2919103820255748, '[N-]': 1.6289874483327829, '\\': 1.5535400325637365, 'Br': 1.5105718048132772, '[NH-]': 1.798927528777345, '6': 1.7293491866176853, '[C@@]': 1.5475291210954323, 'c': 1.0, 'N': 1.2509856983582104, 'Cl': 1.4331804709686848, '[N+]': 1.5167331656034744, '5': 1.594380066179499, '[nH]': 1.4864410689971213, '3': 1.3302593095495208, ')': 1.1336555900413336, '#': 1.4985027709893173, '[NH3+]': 1.545441097578336, 'P': 1.7416132505327901, '[NH2+]': 1.4913053765510265, '[C@H]': 1.368650785106763}


class VAETrainer(TrainerBase):
    def __init__(self, args, vocab, vae, enc_optimizer, dec_optimizer, scheduler, train_bucket_iter, test_bucket_iter):
        super(VAETrainer, self).__init__(args.logdir, args.tag)
        self.logger.info(args)
        self.vocab = vocab
        self.vae = vae
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.scheduler = scheduler

        self.latent_size = args.latent_size
        self.max_len = args.max_len

        self.train_bucket_iter = train_bucket_iter
        self.test_bucket_iter = test_bucket_iter

        print('Train_data_batches length:', len(list(train_bucket_iter)))

        self.loss_weight = None
        if args.weighted_loss:
            print('Train with weighted loss!!!')
            self.loss_weight = []
            for i in range(self.vae.vocab_size):
                token = self.vocab.itos[i]
                if token in weight:
                    self.loss_weight.append(weight[token])
                else:
                    self.loss_weight.append(1.)

        self.loss_function = ops.VAELoss(loss_weight=self.loss_weight)
        self.num_epochs = args.num_epochs
        self.grad_clip = args.grad_clip
        
        self.total_cnt = 0
        self.epoch = 0

    def run_epoch(self, kl_weight=1, epsilon_std=1.0):
        cnt = total_xent_loss = total_kl_loss  = total_loss = 0
        self.vae.train()
        for batch in self.train_bucket_iter:
            kl_weight = ops.kl_anneal_function('linear', step=self.total_cnt, k1=0.1, k2=0.2, max_value=0.1, x0=100000)
            self.tensorboard.add_scalar('train/kl_weight', kl_weight, self.total_cnt)
            self.total_cnt += 1
            x = batch.smile[0]
            lens = batch.smile[1]
            # for target, remove the first <sos> token
            x_target = x[:, 1:].cuda().detach().long()

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            x_hat, z_mean, z_log_var, z = self.vae(x, epsilon_std=epsilon_std)
            xent_loss, kl_loss = self.loss_function.forward(x_target, x_hat, z_mean, z_log_var) 
            
            loss = (xent_loss + kl_weight * kl_loss)
            total_loss += loss.cpu().detach().numpy()
            loss = loss / x.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            total_xent_loss += xent_loss.cpu().detach().numpy()
            total_kl_loss += kl_loss.cpu().detach().numpy()
            cnt += x.size(0)

        return total_loss / cnt, total_xent_loss / cnt, total_kl_loss / cnt
    
    def run_epoch_train(self):
        total_xent_loss = 0
        cnt = 0
        self.vae.train()
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            # for target, remove the first <sos> token
            x_target = x[:, 1:].cuda().detach().long()

            x_hat, z_mean, z_log_var, _ = self.vae(x, epsilon_std=1.0)
            xent_loss, _ = self.loss_function.forward(x_target, x_hat, z_mean, z_log_var)
            total_xent_loss += xent_loss.cpu().detach().numpy()
            cnt += x.size(0)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

        self.tensorboard.add_scalar('train/loss_test_data', total_xent_loss / cnt, self.epoch)
            
    def validate(self, epsilon_std=1.0):
        total_data = reconstructed = reconstructed_valid= 0
        total_bits = reconstructed_bits = 0
        total_loss = 0
        total_mutual_info = 0
        cnt_total = 0
        miss_pred = []
        self.vae.eval()
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            lens = batch.smile[1]
            # calculate mutual information I(x, z)
            mutual_info = self.vae.calc_mi(x)
            total_mutual_info += mutual_info * x.size(0)
            
            z = self.vae.encoder_sample(x, epsilon_std=epsilon_std)[-1]
            smiles_pred, index_pred = self.vae.inference(latent=z, max_len=100)

            # remove start token
            x = x[:, 1:]
            index_pred = index_pred[:, :x.size(1)]
            if index_pred.size(1) < x.size(1):
                padding = torch.LongTensor(np.ones((x.size(0), x.size(1) - index_pred.size(1))))
                index_pred = torch.cat((index_pred, padding), 1)
            # calculate the bit-level accuracy
            reconstructed_bits += torch.sum(x == index_pred).cpu().detach().item()
            total_bits += x.numel()

            total_data += x.size(0)
            for i in range(x.size(0)):
                x_k = x[i].cpu().numpy().tolist()
                x_k = x_k[:(lens[i] - 1)]
                smi_k = [self.vocab.itos[p] for p in x_k]
                smi_k = "".join(smi_k).strip()
                reconstructed += (smi_k == smiles_pred[i])
                reconstructed_valid += len(smiles_pred[i]) > 0
                if smi_k != smiles_pred[i]:
                    miss_pred.append([smi_k, smiles_pred[i]])

            # logits_t = logits_t[:, :x.size(1), :]
            # xent_loss, _ = self.loss_function.forward(x, logits_t, z, z)
            # total_loss += xent_loss.detach().cpu().numpy()
            cnt_total += x.size(0)

        reconstructed_valid = reconstructed_valid / cnt_total
        with open('miss_pred.txt', 'w') as f:
            for line in miss_pred:
                f.write(line[0] + '\n')
                f.write(line[1] + '\n')

        self.tensorboard.add_scalar('test/mutual_info', total_mutual_info / cnt_total, self.epoch)
        self.tensorboard.add_scalar('test/loss', total_loss / cnt_total, self.epoch)
        
        # calculate validity
        n_samples = 2500
        cnt_valid = 0
        cnt_total = 0
        for _ in range(40):
            cnt_total += n_samples
            smiles_pred = self.sample_prior(n_samples=n_samples, max_len=200)
            for smi in smiles_pred:
                if len(smi) > 1 and Chem.MolFromSmiles(smi) is not None:
                    cnt_valid += 1

        self.tensorboard.add_scalar('test/bits_recon_acc', reconstructed_bits / total_bits, self.epoch)
        print('reconstructed_bits_acc: {:.4f}'.format(reconstructed_bits / total_bits))
        print('reconstructed_valid: {:.4f}'.format(reconstructed_valid))
        self.tensorboard.add_scalar('test/validity', cnt_valid / cnt_total, self.epoch)
        print('validity: {:.4f}'.format(cnt_valid / cnt_total))
        return reconstructed / total_data

    def train(self):
        results_fmt = ("{} :: {} :: loss {:.3f} xcent {:.3f} kl {:.3f}" + " " * 30).format
        for self.epoch in range(self.num_epochs):
            epsilon_std = 1.0
            loss, xcent_loss, kl_loss = self.run_epoch(epsilon_std=epsilon_std)
            
            if self.epoch % 10 == 0 and self.epoch >= 20:
                self.save(False, self.epoch)

            self.run_epoch_train()
            self.tensorboard.add_scalar('train/loss', loss, self.epoch)
            self.tensorboard.add_scalar('train/xcent_loss', xcent_loss, self.epoch)
            self.tensorboard.add_scalar('train/kl_loss', kl_loss, self.epoch)
           
            if self.epoch % 10 == 0 and self.epoch >= 20:
                recon_acc = self.validate(epsilon_std=1e-6)
                self.tensorboard.add_scalar('test/recon_acc', recon_acc, self.epoch)
                self.logger.info('recon_acc:' +  str(recon_acc))

            self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, loss, xcent_loss, kl_loss))

    def save(self, is_best, name):
        state = {'vae': self.vae.state_dict(),
                 'enc_optimizer': self.enc_optimizer.state_dict(),
                 'dec_optimizer': self.dec_optimizer.state_dict()}

        path = self.checkpoint_path(name)
        torch.save(state, path)
        if is_best:
            shutil.copyfile(path, self.checkpoint_path('top'))

    def load(self, step):
        self.load_raw(self.checkpoint_path(step))

    def load_raw(self, path):
        state = torch.load(path)
        self.vae.load_state_dict(state['vae'])
        #self.enc_optimizer.load_state_dict(state['enc_optimizer'])
        #self.dec_optimizer.load_state_dict(state['dec_optimizer'])

    def sample_prior(self, n_samples, z_samples=None, max_len=None):
        if z_samples is None:
            z_samples = torch.FloatTensor(n_samples, self.latent_size).normal_(0, 1)
        smiles_pred, _ = self.vae.inference(latent=z_samples.cuda(), max_len=max_len)
        return smiles_pred


class VAEArgParser(TrainerArgParser):
    def __init__(self):
        super(VAEArgParser, self).__init__()
        self.add_argument('--test_mode', action='store_true')
        self.add_argument('--dropout', type=float, default=0.5)
        self.add_argument('--grad_clip', type=float, default=5.0)
        self.add_argument('--wd', type=float, default=1e-4)
        self.add_argument('--batch_size', type=int, default=128)
        self.add_argument('--generate_samples', action='store_true')
        self.add_argument('--weighted_loss', action='store_true')
        self.add_argument('--enc_bidir', action='store_false')
        self.add_argument('--partialsmiles', action='store_true')
        self.add_argument('--n_layers', type=int, default=2)
        self.add_argument('--layer_size', type=int, default=512)
        self.add_argument('--latent_size', type=int, default=56)
        self.add_argument('--embedding_size', type=int, default=48)
        self.add_argument('--max_len', type=int, default=75)
