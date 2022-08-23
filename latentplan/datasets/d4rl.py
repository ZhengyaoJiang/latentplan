import os
import numpy as np
import gym
import random

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

# def construct_dataloader(dataset, **kwargs):
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
#     return dataloader

def minrl_dataset(dataset):
    #dataset = dict(observations=dataset[0]['vector'], actions=dataset[1]['vector'])
    obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []


    trajectory_names = dataset.get_trajectory_names()
    random.shuffle(trajectory_names)

    for trajectory_name in trajectory_names:
        data_gen = dataset.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for obs, action, reward, new_obs, done in data_gen:
            obs = obs['pov'].flatten()
            action = action['vector']

            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done)
            realdone_.append(done)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'rewards': np.array(reward_)[:, None],
        'terminals': np.array(done_)[:, None],
        'realterminals': np.array(realdone_)[:, None],
    }



def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, disable_goal=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate([dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            dataset["observations"] = np.concatenate([dataset["observations"], np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)],
                                                     axis=1)

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] != dataset['infos/goal'][i+1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }
'''

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    #assert 'antmaze' in env.name, 'Using antmaze-specific loading function'


    N = len(dataset['observations'])
    max_episode_length = 701 if 'umaze' in env.name else 1001

    ## set timeouts and terminals to False
    dataset['timeouts'][:] = 0
    dataset['terminals'][:] = 0

    ## fix timeouts
    dataset['timeouts'][max_episode_length-1::max_episode_length] = 1

    fixed = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'steps': [],
    }
    step = 0
    for i in range(N-1):
        done = dataset['terminals'][i] or dataset['timeouts'][i]

        if done:
            ## this is the last observation in its latentplan,
            ## cannot add a next_observation to this transition
            step = 0
            continue

        for key in ['observations', 'actions', 'rewards', 'terminals']:
            val = dataset[key][i]
            fixed[key].append(val)

        next_observation = dataset['observations'][i+1]
        fixed['next_observations'].append(next_observation)

        ## count this as a terminal transition for the purposes of segmentation
        ## iff the next timeout in the raw dataset
        timeout = dataset['timeouts'][i+1]
        fixed['terminals'][-1] += timeout

        fixed['steps'].append(step)

        step += 1

    fixed['rewards'] = np.array(fixed['rewards'])[:,None]
    fixed['terminals'] = np.array(fixed['terminals'])[:,None]
    fixed['steps'] = np.array(fixed['steps'])[:,None]

    fixed = {
        key: np.array(val)
        for key, val in fixed.items()
    }

    ## no termination for the purposes of termination penalty
    fixed['realterminals'] = np.zeros_like(fixed['terminals'])

    max_episode_length = fixed['steps'].max() + 1
    print(f'[ datasets/d4rl ] Max episode length: {max_episode_length}')

    return fixed
'''

class MineRLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return obs['pov'].flatten()

class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return {'vector': action}

    def state_vector(self):
        return np.zeros([1])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps

    env.name = name
    return env
