import os
import numpy as np
import torch
import pdb

import latentplan.utils as utils
import latentplan.datasets as datasets
from latentplan.models.vqvae import TransformerPrior
import wandb


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser(args.savepath)

#######################
####### dataset #######
#######################

env_name = args.dataset if "-v" in args.dataset else args.dataset+"-v0"
env = datasets.load_environment(env_name)

#######################
######## model ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.exp_name,
        'data_config.pkl')
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim+1

representation, _ = utils.load_model(args.logbase, args.dataset, args.exp_name, epoch=args.gpt_epoch, device=args.device)

representation.set_padding_vector(dataset.normalize_joined_single(np.zeros(representation.transition_dim - 1)))

args = Parser().parse_args('train')
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser(args.savepath)
block_size = args.subsampled_sequence_length // args.latent_step
obs_dim = dataset.observation_dim

model_config = utils.Config(
    TransformerPrior,
    savepath=(args.savepath, 'prior_model_config.pkl'),
    ## discretization
    K=args.K, block_size=block_size,
    ## architecture
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd * args.n_head,
    observation_dim=obs_dim,
    ## loss weighting
    latent_step=args.latent_step,
    ## dropout probabilities
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
    obs_shape=args.obs_shape,
)


model = model_config()
model.to(args.device)

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.PriorTrainer,
    savepath=(args.savepath, 'priortrainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    kl_warmup_tokens=warmup_tokens*10,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=0,
    device=args.device,
)

trainer = trainer_config()

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = int(n_epochs // args.n_saves)
wandb.init(project="latentPlanning", config=args, tags=[args.exp_name, args.tag, "prior"])

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

    trainer.train(representation, model, dataset)

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(args.savepath, f'prior_state_{save_epoch}.pt')
    print(f'Saving model to {statepath}')

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)
