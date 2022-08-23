import os
import numpy as np
import torch

import latentplan.utils as utils
import latentplan.datasets as datasets
from latentplan.models.vqvae import VQContinuousVAE
import wandb


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

#######################
######## setup ########
#######################

args = Parser().parse_args('train')

#######################
####### dataset #######
#######################

env_name = args.dataset if "-v" in args.dataset else args.dataset+"-v0"
env = datasets.load_environment(env_name)

sequence_length = args.subsampled_sequence_length * args.step
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser(args.savepath)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)


dataset_class = datasets.SequenceDataset

dataset_config = utils.Config(
    dataset_class,
    savepath=(args.savepath, 'data_config.pkl'),
    env=args.dataset,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    disable_goal=args.disable_goal,
    normalize_raw=args.normalize,
    normalize_reward=args.normalize_reward,
    max_path_length=int(args.max_path_length)
)


dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
if args.task_type == "locomotion":
    transition_dim = obs_dim+act_dim+3
else:
    transition_dim = 128+act_dim+3

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim # total number of dimensionalities for a maximum length sequence (T)

print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)



model_config = utils.Config(
    VQContinuousVAE,
    savepath=(args.savepath, 'model_config.pkl'),
    ## discretization
    vocab_size=args.N, block_size=block_size,
    K=args.K,
    ## architecture
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd * args.n_head,
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## loss weighting
    action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
    position_weight=args.position_weight,
    first_action_weight=args.first_action_weight,
    sum_reward_weight=args.sum_reward_weight,
    last_value_weight=args.last_value_weight,
    trajectory_embd=args.trajectory_embd,
    model=args.model,
    latent_step=args.latent_step,
    ma_update=args.ma_update,
    residual=args.residual,
    obs_shape=args.obs_shape,
    ## dropout probabilities
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
    bottleneck=args.bottleneck,
    masking=args.masking,
    state_conditional=args.state_conditional,
)


model = model_config()
model.to(args.device)
if args.normalize:
    model.set_padding_vector(dataset.normalize_joined_single(np.zeros(model.transition_dim-1)))

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.VQTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
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
wandb.init(project="latentPlanning", config=args, tags=[args.exp_name, args.tag])

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

    trainer.train(model, dataset)

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
    print(f'Saving model to {statepath}')

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)
