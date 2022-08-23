import math
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import wandb

from .timer import Timer

def to(xs, device):
    return [x.to(device) for x in xs]

class VQTrainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        for _ in range(n_epochs):

            losses = []
            timer = Timer()
            for it, batch_numpy in enumerate(loader):
                batch = to(batch_numpy, self.device)

                # decay the learning rate based on our progress
                y = batch[-2]
                self.n_tokens += np.prod(y.shape)
                if self.n_tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.n_tokens - config.warmup_tokens) / float(
                        max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

                if config.lr_decay:
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate


                # forward the model
                with torch.set_grad_enabled(True):
                    *_, recon_loss, vq_loss, commit_loss = model(*batch)
                    loss = (recon_loss+vq_loss+commit_loss).mean()
                    losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # report progress
                if it % log_freq == 0:
                    if dataset.test_portion == 0:
                        summary = dict(recontruction_loss=recon_loss.item(),
                                       commit_loss=commit_loss.item(),
                                       lr=lr,
                                       lr_mulr=lr_mult,
                                       )
                        print(
                            f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                            f'train reconstruction loss {recon_loss.item():.5f} | '
                            f' train commit loss {commit_loss.item():.5f} |'
                            f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                            f't: {timer():.2f}')
                    else:
                        torch.cuda.empty_cache()
                        model.eval()
                        with torch.set_grad_enabled(False):
                            _, t_recon_loss, t_vq_loss, t_commit_loss = model(*to(dataset.get_test(), self.device))
                        model.train()
                        summary = dict(recontruction_loss=recon_loss.item(),
                                       commit_loss=commit_loss.item(),
                                       test_reconstruction_loss= t_recon_loss.item(),
                                       lr=lr,
                                       lr_mulr=lr_mult,
                                       )
                        print(
                            f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                            f'train reconstruction loss {recon_loss.item():.5f} |'
                            f' train commit loss {commit_loss.item():.5f} |'
                            f' test reconstruction loss {t_recon_loss.item():.5f} |'
                            f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                            f't: {timer():.2f}')
                    wandb.log(summary, step=self.n_epochs*len(loader)+it)
                if dataset.test_portion >= 0:
                    torch.cuda.empty_cache()
            self.n_epochs += 1


class PriorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, representation, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        representation.train(False)
        model.train(True)

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        for _ in range(n_epochs):

            losses = []
            timer = Timer()
            for it, batch_numpy in enumerate(loader):
                batch = to(batch_numpy, self.device)

                # decay the learning rate based on our progress
                y = batch[1]
                self.n_tokens += np.prod(y.shape)
                if self.n_tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.n_tokens - config.warmup_tokens) / float(
                        max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

                if config.lr_decay:
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                states = batch[0][:, 0, :model.observation_dim]
                indices = representation.encode(batch[0], batch[-1])

                # forward the model
                with torch.set_grad_enabled(True):
                    _, loss = model(indices[:, :-1], states, indices)
                    losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # report progress
                if it % log_freq == 0:
                    summary = dict(loss=loss.item(),
                                   lr=lr,
                                   lr_mulr=lr_mult, )
                    print(
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f' train loss {loss.item():.5f} |'
                        f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')
                    wandb.log(summary, step=self.n_epochs * len(loader) + it)
            self.n_epochs += 1