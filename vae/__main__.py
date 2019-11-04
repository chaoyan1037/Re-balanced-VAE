
import tqdm
import os
import sys
sys.path.append(".")

import torch
import time
from rdkit import Chem

from base_classes import schedulers
from data.molecule_iterator import SmileBucketIterator
from vae import vae_models
from vae.vae_trainer import VAETrainer, VAEArgParser


if __name__ == '__main__':
    args = VAEArgParser().parse_args()
    print(args)
    
    smi_file = 'data/zinc.smi'
    vocab_file = 'data/vacab'
    smi_iterator = SmileBucketIterator(smi_file, vocab_file, args.batch_size)
    
    if args.test_mode:
        smi_iterator.train_smi.examples = smi_iterator.train_smi.examples[:1000]

    train_bucket_iter = smi_iterator.train_bucket_iter()
    test_bucket_iter = smi_iterator.test_bucket_iter()
    vocab_size = smi_iterator.vocab_size
    padding_idx = smi_iterator.padding_idx
    sos_idx = smi_iterator.sos_idx
    eos_idx = smi_iterator.eos_idx
    unk_idx = smi_iterator.unk_idx
    vocab = smi_iterator.get_vocab()
    print('vocab_size:', vocab_size)
    print('padding_idx sos_idx eos_idx unk_idx:', padding_idx, sos_idx, eos_idx, unk_idx)
    print(vocab.itos, vocab.stoi)

    # define Vae model
    vae = vae_models.Vae(vocab, vocab_size, args.embedding_size, args.dropout,
                         padding_idx, sos_idx, unk_idx,
                         args.max_len, args.n_layers, args.layer_size,
                         bidirectional=args.enc_bidir, latent_size=args.latent_size, partialsmiles=args.partialsmiles)
    
    enc_optimizer = torch.optim.Adam(vae.encoder_params, lr=3e-4)
    dec_optimizer = torch.optim.Adam(vae.decoder_params, lr=1e-4)

    #scheduler = schedulers.Scheduler(enc_optimizer, dec_optimizer, 0.5, 1e-8)
    scheduler = schedulers.StepScheduler(enc_optimizer, dec_optimizer, epoch_anchors=[200, 250, 275])
    vae = vae.cuda()
    trainer = VAETrainer(args, vocab, vae, enc_optimizer, dec_optimizer, scheduler, train_bucket_iter, test_bucket_iter)

    if args.generate_samples:
        if not args.restore:
            raise ValueError('argument --restore with trained vae path required to generate samples!')        trainer.load_raw(args.restore)

        # random sampling
        samples = []
        for _ in tqdm.tqdm(range(10)):
            samples.append(trainer.sample_prior(1000).cpu())
        samples = torch.cat(samples, 0)
        torch.save(samples, 'prior_samples.pkl')
    else:
        # validate
        if args.restore:
            trainer.load_raw(args.restore)
        else:
            trainer.train()
        print('recon_acc:', trainer.validate(epsilon_std=1e-6))

