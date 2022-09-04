Code for our paper **[Re-balancing Variational Autoencoder Loss for Molecule Sequence Generation](https://arxiv.org/pdf/1910.00698.pdf)** . 

We use pytorch 0.4.1 and python 3.

Run the training by:

```
python vae
```

The default logdir is */mnt/Data/DL/tmp/beta_vae_mol*， which is specified in the file */base_classes/trainer.py*.

After training, you can run the testing by:
```
python vae --restore=path_to_checkpoint_dir/150
```

Thanks for the **[partialsmiles](https://github.com/baoilleach/partialsmiles)** package, we can guide the molecule sequence generataion now. Enable guiding by the **partialsmiles**
in generation procedure by:
```
python vae --restore=path_to_checkpoint_dir/150 --partialsmiles
```

Our implementation is modified on the code from https://github.com/DavidJanz/molecule_grammar_rnn.

If you find our code useful, please cite our paper:

@inproceedings{yan2020re,<br />
  title={Re-balancing variational autoencoder loss for molecule sequence generation},<br />
  author={Yan, Chaochao and Wang, Sheng and Yang, Jinyu and Xu, Tingyang and Huang, Junzhou},<br />
  booktitle={Proceedings of the 11th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},<br />
  pages={1--7},<br />
  year={2020}
}

@article{yan2022molecule,<br />
  title={Molecule Sequence Generation with Rebalanced Variational Autoencoder Loss},<br />
  author={Yan, Chaochao and Yang, Jinyu and Ma, Hehuan and Wang, Sheng and Huang, Junzhou},<br />
  journal={Journal of Computational Biology},<br />
  year={2022},<br />
  publisher={Mary Ann Liebert}
}
