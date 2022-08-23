import numpy as np
import torch
import pdb

from ..utils.arrays import to_torch

VALUE_PLACEHOLDER = 1e6

def make_prefix(obs, transition_dim, device="cuda"):
    obs_discrete = to_torch(obs, dtype=torch.float32, device=device)
    pad_dims = to_torch(np.zeros(transition_dim - len(obs)), dtype=torch.float32, device=device)
    if obs_discrete.ndim == 1:
        obs_discrete = obs_discrete.reshape(1, 1, -1)
        pad_dims = pad_dims.reshape(1, 1, -1)
    transition = torch.cat([obs_discrete, pad_dims], axis=-1)
    prefix = transition
    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def extract_actions_continuous(x, observation_dim, action_dim, t=None):
    assert x.shape[0] == 1
    actions = x[0, :, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(observation, action, reward, device):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])
    context = []

    transition_discrete = to_torch(transition, dtype=torch.float32, device=device)
    transition_discrete = transition_discrete.reshape(1, 1, -1)

    ## add new transition to context
    context.append(transition_discrete)
    return context
