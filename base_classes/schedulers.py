
class OptimiserFailed(Exception):
    pass

class StepScheduler:
    def __init__(self, enc_optimizer, dec_optimizer, lr_step=0.5, epoch_anchors=[200, 250, 275]):
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self._lr_step = lr_step
        self._epoch_anchors = epoch_anchors

    def step_lr(self, current_epoch):
        if current_epoch in self._epoch_anchors:
            self.lr = self._lr_step * self.lr

    @property
    def lr(self):
        for pg in self.enc_optimizer.param_groups:
            lr = pg['lr']
        return lr

    @lr.setter
    def lr(self, new_lr):
        for pg in self.enc_optimizer.param_groups:
            pg['lr'] = new_lr
        for pg in self.dec_optimizer.param_groups:
            pg['lr'] = new_lr

    def state_dict(self):
        return {'optim_enc_state_dict': self.enc_optimizer.state_dict(),
                'optim_dec_state_dict': self.dec_optimizer.state_dict(),
                'lr_step': self._lr_step,
                'epoch_anchors': self._epoch_anchors}

    def load_state_dict(self, state_dict):
        self.enc_optimizer.load_state_dict(state_dict['optim_enc_state_dict'])
        self.dec_optimizer.load_state_dict(state_dict['optim_dec_state_dict'])
        self._lr_step = state_dict['lr_step']
        self._epoch_anchors = state_dict['epoch_anchors']


class Scheduler:
    def __init__(self, enc_optimizer, dec_optimizer, lr_step, lr_min, patience=250, rtol=0.5, atol=0.1, eps=1e-2):
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self._steps = 0
        self.top_obj = float('inf')

        self._rtol = rtol
        self._atol = atol

        self.patience = patience
        self._patience = patience
        self._lr_step = lr_step
        self._lr_min = lr_min
        self._eps = eps
        self.is_best = False

    def step(self, objective):
        self._steps += 1
        if objective < self.top_obj - self._eps:
            self.top_obj = objective
            self.is_best = True
            self._patience = self.patience
        else:
            self.is_best = False
            # if objective > self.top_obj * (1 + self._rtol) + self._atol:
            #     raise OptimiserFailed()
            # else:
            self._patience -= 1

        if self._patience == 0:
            self.lr *= self._lr_step
            self._patience = self.patience

    def step_lr(self):
        self.lr = self._lr_step(self.lr)

    @property
    def lr(self):
        for pg in self.enc_optimizer.param_groups:
            lr = pg['lr']
        return lr

    @lr.setter
    def lr(self, new_lr):
        if new_lr < self._lr_min:
            pass
        else:
            for pg in self.enc_optimizer.param_groups:
                pg['lr'] = new_lr
            for pg in self.dec_optimizer.param_groups:
                pg['lr'] = new_lr

    def state_dict(self):
        return {'optim_enc_state_dict': self.enc_optimizer.state_dict(),
                'optim_dec_state_dict': self.dec_optimizer.state_dict(),
                'steps': self._steps,
                'top_obj': self.top_obj}

    def load_state_dict(self, state_dict):
        self.enc_optimizer.load_state_dict(state_dict['optim_enc_state_dict'])
        self.dec_optimizer.load_state_dict(state_dict['optim_dec_state_dict'])
        self._steps = state_dict['steps']
        self.top_obj = state_dict['top_obj']
