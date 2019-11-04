import argparse
import logging
import os
import sys
import time

import tensorboardX


class TrainerBase:
    def __init__(self, logdir, tag):
        self.log_path = TrainerBase.log_path(logdir, tag)
        os.makedirs(self.log_path, exist_ok=True)

        self.checkpoint_base_path = os.path.join(self.log_path, 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.logger = self.setup_logging(self.log_path)
        self.tensorboard = tensorboardX.SummaryWriter(log_dir=self.log_path)

    def checkpoint_path(self, step):
        return os.path.join(self.checkpoint_base_path, str(step))

    @staticmethod
    def log_path(logdir, tag):
        time_str = time.strftime("%d-%H:%M:%S", time.localtime())
        fname = "{}_{}".format(tag, time_str) if tag is not None else time_str
        return os.path.join(logdir, fname)

    @staticmethod
    def setup_logging(logpath):
        logger = logging.getLogger()
        logger.propagate = False
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s::%(message)s')

        hdlr = logging.FileHandler(os.path.join(logpath, 'out.log'))
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        hdlr = logging.StreamHandler(stream=sys.stdout)
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        return logger


class TrainerArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(TrainerArgParser, self).__init__()
        self.add_argument('--logdir', type=str, default='/mnt/Data/DL/tmp/beta_vae_mol')
        self.add_argument('--restore', type=str, default='')
        self.add_argument('--tag', type=str, default='')
        self.add_argument('--num_epochs', type=int, default=151)
