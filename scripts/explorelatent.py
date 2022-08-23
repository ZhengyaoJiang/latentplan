import json
import pdb
from os.path import join
import os
import numpy as np

import latentplan.utils as utils
import latentplan.datasets as datasets
import torch

from latentplan.search import (
    sample_with_prior,
    beam_with_prior,
    make_prefix,
    extract_actions,
    update_context,
)

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'


def visualize_from_latent(args, indicies, prefix, model, denormalize_rew, denormalize_val, discount, opt="max"):
    with torch.no_grad():
        prediction_raw = model.decode_from_indices(indicies, prefix[:, 0, :model.observation_dim])
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([indicies.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([indicies.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:, :-1] * discounts[:, :-1], dim=-1) + V_t[:, -1] * discounts[:, -1]
    if opt=="max":
        optimal = prediction_raw[values.argmax()]
        print(f"value {values.max().item()}")
    else:
        optimal = prediction_raw[values.argmin()]
        print(f"value {values.min().item()}")
    if dataset.normalized_raw:
        sequence_recon = dataset.denormalize_joined(optimal.cpu().numpy())
    else:
        sequence_recon = optimal.cpu().numpy()


    vidoes = renderer.render_plan(join(args.savepath, f'{t}_plan-{indicies.cpu().numpy()}.mp4'), sequence_recon, state)

    print("finished")


#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser("/tmp/explorelatent")
args.nb_samples = int(args.nb_samples)
args.n_expand = int(args.n_expand)
args.beam_width = int(args.beam_width)
args.horizon = int(args.horizon)
args.rounds = int(args.rounds)

#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.exp_name,
        'data_config.pkl')


#######################
####### dataset #######
#######################

for e in range(0, 10):
    args.exp_name = f"T-58-{1}"

    model, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.exp_name,
                                        epoch=args.gpt_epoch, device=args.device)

    prior, _ = utils.load_prior_model(args.logbase, args.dataset, args.exp_name,
                                      epoch=args.gpt_epoch, device=args.device)

    model.set_padding_vector(dataset.normalize_joined_single(np.zeros(model.transition_dim - 1)))

    env = datasets.load_environment(args.dataset)
    renderer = utils.make_renderer(args)
    timer = utils.timer.Timer()

    if args.discrete:
        discretizer = dataset.discretizer
    else:
        discretizer = None

    value_fn = None
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
    if "antmaze" in env.name:
        observation = np.concatenate([observation, env.target_goal])
    total_reward = 0

    ## observations for rendering
    rollout = [observation.copy()]

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    T = env.max_episode_steps
    model.eval()
    for t in range(T):

        state = env.state_vector()
        observation = preprocess_fn(observation)
        if dataset.normalized_raw:
            observation = dataset.normalize_states(observation)

        if "antmaze" in env.name:
            state = np.concatenate([state, env.target_goal])

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, transition_dim, args.prefix_context)[-1, -1, None, None]
            #sequence = sample(model, prefix, denormalize_rew=dataset.denormalize_rewards,
            #                  denormalize_val=dataset.denormalize_values,
            #                  steps=args.horizon - args.max_context_transitions - 1,
            #                  discount=discount)
            if args.test_planner == "beam_prior":
                prior.eval()
                info = None
                sequence, info_beam = beam_with_prior(prior, value_fn, model, prefix, denormalize_rew=dataset.denormalize_rewards,
                                           denormalize_val=dataset.denormalize_values,
                                           steps=int(args.horizon) - args.max_context_transitions - 1,
                                           beam_width=args.beam_width,
                                           n_expand=args.n_expand,
                                           likelihood_weight=args.prob_weight,
                                           prob_threshold=float(args.prob_threshold),
                                           discount=discount,
                                           return_info=True)
            elif args.test_planner == "sample_prior":
                prior.eval()
                info_beam = None
                sequence, info = sample_with_prior(prior, value_fn, model, prefix, denormalize_rew=dataset.denormalize_rewards,
                                                   denormalize_val=dataset.denormalize_values,
                                                   steps=int(args.horizon) - args.max_context_transitions - 1,
                                                   nb_samples=2048,
                                                   rounds=64,
                                                   prob_threshold=0.05,
                                                   likelihood_weight=5e2,
                                                   uniform=False,
                                                   discount=discount,
                                                   return_info=True)
            else:
                raise ValueError(f"unknown planner {args.test_planner}")
        else:
            sequence = sequence[1:]

        print(dataset.denormalize_values(sequence[0,-2]))

        ## [ horizon x transition_dim ] convert sampled tokens to continuous latentplan
        if args.discrete:
            sequence_recon = discretizer.reconstruct(sequence)
        else:
            sequence_recon = sequence

        ## [ action_dim ] index into sampled latentplan to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        if dataset.normalized_raw:
            action = dataset.denormalize_actions(action)
            sequence_recon = dataset.denormalize_joined(sequence_recon)
            if info:
                shape = info["predictions"].shape
                predictions = info["predictions"].reshape([-1, shape[-1]])
                info["predictions"] = dataset.denormalize_joined(predictions).reshape(shape)
            else:
                for step, info_step in info_beam.items():
                    shape = info_step["predictions"].shape
                    predictions = info_step["predictions"].reshape([-1, shape[-1]])
                    info_step["predictions"] = dataset.denormalize_joined(predictions).reshape(shape)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        if "antmaze" in env.name:
            next_observation = np.concatenate([next_observation, env.target_goal])

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions, device=args.device)

        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
            f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
        )
        ## visualization
        if t % 100 == 0:
            nb_latent = (args.horizon - args.max_context_transitions - 1) // model.latent_step
            import pickle

            prior.eval()
            mses = []
            if info:
                for prediction in info["predictions"]:
                    mses.append(renderer.compute_mse(prediction, state))
                info["mse"] = np.array(mses)

                with open(f"./analysis/samples_step{e}-{t}.pkl", "wb") as f:
                    pickle.dump(info, f)
            else:
                with open(f"./analysis/samples_step{e}-{t}.pkl", "wb") as f:
                    pickle.dump(info_beam, f)

            '''
            while True:
                opt="max"
                command = input(f"input command, r for random index, {nb_latent} numbers for manually specified latents,"
                                f" q for quit")
                if command in ["r", "random"]:
                    indicies = torch.randint(0, model.model.K - 1, size=[1, nb_latent],
                                             device=prefix.device, dtype=torch.int32)
                elif command in ["q", "quit"]:
                    break
                elif command in ["s", "sameple_save"]:
                    
                    break
                elif "sample" in command:
                    if command == "sample_min":
                        opt = "min"
                    elif command == "sample_max":
                        pass
                    else:
                        raise ValueError()
                    indicies = torch.randint(0, model.model.K - 1, size=[4096, nb_latent],
                                             device=prefix.device, dtype=torch.int32)
                else:
                    numbers = [int(s) for s in command.split(" ")]
                    indicies = torch.tensor(numbers, device=prefix.device, dtype=torch.int32)[None, :]
                visualize_from_latent(args, indicies, prefix, model, denormalize_rew=dataset.denormalize_rewards,
                              denormalize_val=dataset.denormalize_values, discount=discount, opt=opt)
            '''
        if terminal: break

        observation = next_observation

    ## save result as a json file
    #json_path = join(args.savepath, 'rollout.json')
    #json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
