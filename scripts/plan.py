import json
from os.path import join
import os
import numpy as np

import latentplan.utils as utils
import latentplan.datasets as datasets
from latentplan.search import (
    enumerate_all,
    sample_with_prior,
    sample_with_prior_tree,
    beam_with_prior,
    beam_with_uniform,
    beam_mimic,
    make_prefix,
    extract_actions,
    update_context,
)

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
args.nb_samples = int(args.nb_samples)
args.n_expand = int(args.n_expand)
args.beam_width = int(args.beam_width)
args.horizon = int(args.horizon)
args.rounds = int(args.rounds)
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser(args.savepath)
args.uniform = bool(args.uniform)
try:
    args.prob_weight = float(args.prob_weight)
except:
    args.prob_weight = 5e2


#######################
####### models ########
#######################

env = datasets.load_environment(args.dataset)
dataset = utils.load_from_config(args.logbase, args.dataset, args.exp_name,
        'data_config.pkl')


gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.exp_name,
        epoch=args.gpt_epoch, device=args.device)
#gpt.reset_model()
#gpt.to('cuda')

if args.test_planner in ["sample_prior", "sample_prior_tree", "beam_prior", "beam_mimic", "beam_uniform"]:
    prior, _ = utils.load_prior_model(args.logbase, args.dataset, args.exp_name,
                                      epoch=args.gpt_epoch, device=args.device)
#prior.reset_model()


gpt.set_padding_vector(dataset.normalize_joined_single(np.zeros(gpt.transition_dim-1)))
#######################
####### dataset #######
#######################

if args.task_type == "locomotion":
    renderer = utils.make_renderer(args)
timer = utils.timer.Timer()

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################
REWARD_DIM = VALUE_DIM = 1
transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
observation = env.reset()
total_reward = 0
discount_return = 0

if "antmaze" in env.name:
    if dataset.disable_goal:
        observation = np.concatenate([observation, np.zeros([2], dtype=np.float32)])
        rollout = [np.concatenate([env.state_vector().copy(), np.zeros([2], dtype=np.float32)])]
    else:
        observation = np.concatenate([observation, env.target_goal])
        rollout = [np.concatenate([env.state_vector().copy(), env.target_goal])]
else:
    rollout = [np.concatenate([env.state_vector().copy()])]

## previous (tokenized) transitions for conditioning transformer
context = []
mses = []

T = env.max_episode_steps
gpt.eval()
for t in range(T):

    observation = preprocess_fn(observation)
    state = env.state_vector()


    if dataset.normalized_raw:
        observation = dataset.normalize_states(observation)

    if "antmaze" in env.name:
        if dataset.disable_goal:
            state = np.concatenate([state, np.zeros([2], dtype=np.float32)])
        else:
            state = np.concatenate([state, env.target_goal])

    if t % args.plan_freq == 0:
        ## concatenate previous transitions and current observations to input to model
        prefix = make_prefix(observation, transition_dim, device=args.device)[-1, -1, None, None]

        ## sample sequence from model beginning with `prefix`
        if args.test_planner == 'beam_prior':
            prior.eval()
            sequence = beam_with_prior(prior, gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                              denormalize_val=dataset.denormalize_values,
                              steps=int(args.horizon),
                              beam_width=args.beam_width,
                              n_expand=args.n_expand,
                              likelihood_weight=args.prob_weight,
                              prob_threshold=float(args.prob_threshold),
                              discount=discount)
        elif args.test_planner == 'beam_uniform':
            prior.eval()
            sequence = beam_with_uniform(prior, gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                                       denormalize_val=dataset.denormalize_values,
                                       steps=int(args.horizon),
                                       beam_width=args.beam_width,
                                       n_expand=args.n_expand,
                                       prob_threshold=float(args.prob_threshold),
                                       discount=discount)
        elif args.test_planner == 'beam_mimic':
            prior.eval()
            sequence = beam_mimic(prior, gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                                       denormalize_val=dataset.denormalize_values,
                                       steps=int(args.horizon),
                                       beam_width=args.beam_width,
                                       n_expand=args.n_expand,
                                       prob_threshold=float(args.prob_threshold),
                                       discount=discount)
        elif args.test_planner == "enumerate":
            sequence = enumerate_all(gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                                     denormalize_val=dataset.denormalize_values,
                                     discount=discount)
        elif args.test_planner == "sample_prior":
            prior.eval()
            sequence = sample_with_prior(prior, gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                              denormalize_val=dataset.denormalize_values,
                              steps=int(args.horizon),
                              nb_samples=args.nb_samples,
                              rounds=args.rounds,
                              prob_threshold=float(args.prob_threshold),
                              likelihood_weight=args.prob_weight,
                              uniform=args.uniform,
                              discount=discount)
        elif args.test_planner == "sample_prior_tree":
            prior.eval()
            sequence = sample_with_prior_tree(prior, gpt, prefix, denormalize_rew=dataset.denormalize_rewards,
                                         denormalize_val=dataset.denormalize_values,
                                         steps=int(args.horizon) - args.max_context_transitions - 1,
                                         discount=discount)
    else:
        sequence = sequence[1:]

    if t == 0:
        first_value = float(dataset.denormalize_values(sequence[0,-2]))
        first_search_value = float(dataset.denormalize_values(sequence[-1, -2]))
    print(dataset.denormalize_values(sequence[0,-2]))

    ## [ horizon x transition_dim ] convert sampled tokens to continuous latentplan
    sequence_recon = sequence

    ## [ action_dim ] index into sampled latentplan to grab first action

    feature_dim = dataset.observation_dim
    action = extract_actions(sequence_recon, feature_dim, action_dim, t=0)
    if dataset.normalized_raw:
        action = dataset.denormalize_actions(action)
        sequence_recon = dataset.denormalize_joined(sequence_recon)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    if "antmaze" in env.name:
        if dataset.disable_goal:
            next_observation = np.concatenate([next_observation, np.zeros([2], dtype=np.float32)])
        else:
            next_observation = np.concatenate([next_observation, env.target_goal])


    ## update return
    total_reward += reward
    discount_return += reward* discount**(t)
    score = env.get_normalized_score(total_reward)

    rollout.append(state.copy())
    context = update_context(observation, action, reward, device=args.device)

    print(
        f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
        f'time: {timer():.4f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
    )

    ## visualization
    if t % args.vis_freq == 0 or terminal or t == T-1:
        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)

        # ffmpeg will report a error in some setup
        if "antmaze" in env.name or "medium" in env.name:
            _, mse = renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'),
                             sequence_recon, state)
        else:
            _, mse = renderer.render_real(join(args.savepath, f'{t}_plan.mp4'),
                                          sequence_recon, state)

        ## save rollout thus far
        renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
        if not terminal:
            mses.append(mse)

    if terminal: break

    observation = next_observation

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch,
             'first_value': first_value, 'first_search_value': first_search_value, 'discount_return': discount_return,
             'prediction_error': np.mean(mses)}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
