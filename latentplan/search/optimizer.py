from collections import defaultdict
import torch


REWARD_DIM = VALUE_DIM = 1

@torch.no_grad()
def model_rollout_continuous(model, x, latent, denormalize_rew, denormalize_val, discount, prob_penalty_weight=1e4):
    prediction = model.decode(latent, x[:, -1, :model.observation_dim])
    prediction = prediction.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([x.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([x.shape[0], -1])

    # discounts with terminal flag
    terminal = prediction[:, -1].reshape([x.shape[0], -1])
    discounts = torch.cumprod(torch.ones_like(r_t) * discount * (1-terminal), dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
    prob_penalty = prob_penalty_weight * torch.mean(torch.square(latent), dim=-1)
    objective = values - prob_penalty
    return objective.cpu().numpy(), prediction.cpu().numpy()


import numpy as np


@torch.no_grad()
def sample(model, x, denormalize_rew, denormalize_val, discount, steps, nb_samples=4096, rounds=8):
    indicies = torch.randint(0, model.model.K-1, size=[nb_samples, steps // model.latent_step],
                             device=x.device, dtype=torch.int32)
    prediction_raw = model.decode_from_indices(indicies, x[:, 0, :model.observation_dim])
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([indicies.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([indicies.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
    optimal = prediction_raw[values.argmax()]
    print(values.max().item())
    return optimal.cpu().numpy()


@torch.no_grad()
def sample_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps, nb_samples=4096, rounds=8,
                      likelihood_weight=5e2, prob_threshold=0.05, uniform=False, return_info=False):
    state = x[:, 0, :model.observation_dim]
    optimals = []
    optimal_values = []
    info = defaultdict(list)
    for round in range(rounds):
        contex = None
        acc_probs = torch.zeros([1]).to(x)
        for step in range(steps//model.latent_step):
            logits, _ = prior(contex, state) # [B x t x K]
            probs = raw_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
            log_probs = torch.log(probs)
            if uniform:
                valid = probs > 0
                probs = valid/valid.sum(dim=-1)[:, None]
            if step == 0:
                samples = torch.multinomial(probs, num_samples=nb_samples//rounds, replacement=True) # [B, M]
            else:
                samples = torch.multinomial(probs, num_samples=1, replacement=True)  # [B, M]
            samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)]) # [B, M]
            acc_probs = acc_probs + samples_prob.reshape([-1])
            if not contex is None:
                contex = torch.cat([contex, samples.reshape([-1, 1])], dim=1)
            else:
                contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        prediction_raw = model.decode_from_indices(contex, state)
        prediction = prediction_raw.reshape([-1, model.transition_dim])

        r_t = prediction[:, -3]
        V_t = prediction[:, -2]
        terminals = prediction[:, -1].reshape([contex.shape[0], -1])
        if denormalize_rew is not None:
            r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
        if denormalize_val is not None:
            V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

        discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
        values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
        likelihood_bonus = likelihood_weight*torch.clamp(acc_probs, -1e5, np.log(prob_threshold)*(steps//model.latent_step))
        info["log_probs"].append(acc_probs.cpu().numpy())
        info["returns"].append(values.cpu().numpy())
        info["predictions"].append(prediction_raw.cpu().numpy())
        info["objectives"].append(values.cpu().numpy() + likelihood_bonus.cpu().numpy())
        info["latent_codes"].append(contex.cpu().numpy())
        max_idx = (values+likelihood_bonus).argmax()
        optimal_value = values[max_idx]
        optimal = prediction_raw[max_idx]
        optimals.append(optimal)
        optimal_values.append(optimal_value.item())

    for key, val in info.items():
        info[key] = np.concatenate(val, axis=0)
    max_idx = np.array(optimal_values).argmax()
    optimal = optimals[max_idx]
    print(f"predicted max value {optimal_values[max_idx]}")
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()


@torch.no_grad()
def sample_with_prior_tree(prior, model, x, denormalize_rew, denormalize_val, discount, steps, samples_per_latent=16, likelihood_weight=0.0):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        samples = torch.multinomial(probs, num_samples=samples_per_latent, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(samples_per_latent, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, samples_per_latent, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

    prediction_raw = model.decode_from_indices(contex, state)
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]


    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
    likelihood_bouns = likelihood_weight*torch.log(acc_probs)
    max_idx = (values+likelihood_bouns).argmax()
    optimal = prediction_raw[max_idx]
    print(f"predicted max value {values[max_idx]}, likelihood {acc_probs[max_idx]} with bouns {likelihood_bouns[max_idx]}")
    return optimal.cpu().numpy()

@torch.no_grad()
def beam_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=False):
    contex = None
    state = x[:, 0, :prior.observation_dim]
    acc_probs = torch.zeros([1]).to(x)
    info = {}
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        #logits = logits[:, -1, :]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        log_probs = torch.log(probs)
        nb_samples = beam_width * n_expand if step == 0 else n_expand
        samples = torch.multinomial(probs, num_samples=nb_samples, replacement=True) # [B, M]
        samples_log_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)]) # [B, M]
        if prob_acc in ["product", "expect"]:
            acc_probs = acc_probs.repeat_interleave(nb_samples, 0) + samples_log_prob.reshape([-1])
        elif prob_acc == "min":
            acc_probs = torch.minimum(acc_probs.repeat_interleave(nb_samples, 0), samples_log_prob.reshape([-1]))
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        prediction_raw = model.decode_from_indices(contex, state)
        prediction = prediction_raw.reshape([-1, model.action_dim+model.observation_dim+3])
        #terminals = prediction[:, -1].reshape([contex.shape[0], -1])
        # shift to right 1 step
        #terminals = torch.cat([torch.zeros([contex.shape[0], 1], device=terminals.device), terminals[:, :-1]], dim=-1)

        r_t = prediction[:, -3]
        V_t = prediction[:, -2]

        if denormalize_rew is not None:
            r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
        if denormalize_val is not None:
            V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

        discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
        values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
        if prob_acc == "product":
            likelihood_bonus = likelihood_weight*torch.clamp(acc_probs, -1e5, np.log(prob_threshold)*(steps//model.latent_step))
        elif prob_acc == "min":
            likelihood_bonus = likelihood_weight*torch.clamp(acc_probs, 0, np.log(prob_threshold))
        nb_top = beam_width if step < (steps//model.latent_step-1) else 1
        if prob_acc == "expect":
            values_with_b, index = torch.topk(values*torch.exp(acc_probs), nb_top)
        else:
            values_with_b, index = torch.topk(values+likelihood_bonus, nb_top)
        if return_info:
            info[(step+1)*model.latent_step] = dict(predictions=prediction_raw.cpu(), returns=values.cpu(),
                                                    latent_codes=contex.cpu(), log_probs=acc_probs.cpu(),
                                                    objectives=values+likelihood_bonus, index=index.cpu())
        contex = contex[index]
        acc_probs = acc_probs[index]

    optimal = prediction_raw[index[0]]
    print(f"predicted max value {values[0]}")
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()

@torch.no_grad()
def beam_with_uniform(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand,  prob_threshold=0.05):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width * n_expand if step == 0 else n_expand
        valid = probs > prob_threshold
        samples = torch.multinomial(valid/valid.sum(dim=-1), num_samples=nb_samples, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(nb_samples, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        prediction_raw = model.decode_from_indices(contex, state)
        prediction = prediction_raw.reshape([-1, model.transition_dim])
        r_t, V_t = prediction[:, -3], prediction[:, -2]

        if denormalize_rew is not None:
            r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
        if denormalize_val is not None:
            V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])


        discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
        values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
        nb_top = beam_width if step < (steps//model.latent_step-1) else 1
        values, index = torch.topk(values, nb_top)
        contex = contex[index]
        acc_probs = acc_probs[index]

    optimal = prediction_raw[index[0]]
    print(f"predicted max value {values[0]}")
    return optimal.cpu().numpy()

@torch.no_grad()
def beam_mimic(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand,  prob_threshold=0.05):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width * n_expand if step == 0 else n_expand
        samples = torch.multinomial(probs, num_samples=nb_samples, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(nb_samples, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        nb_top = beam_width if step < (steps//model.latent_step-1) else 1
        values, index = torch.topk(acc_probs, nb_top)
        contex = contex[index]
        acc_probs = acc_probs[index]

    prediction_raw = model.decode_from_indices(contex, state)
    optimal = prediction_raw[0]
    print(f"value {values[0]}, prob {acc_probs[0]}")
    return optimal.cpu().numpy()


@torch.no_grad()
def enumerate_all(model, x, denormalize_rew, denormalize_val, discount):
    indicies = torch.range(0, model.model.K-1, device=x.device, dtype=torch.int32)
    prediction_raw = model.decode_from_indices(indicies, x[:, 0, :model.observation_dim])
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -2], prediction[:, -1]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([indicies.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([indicies.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
    optimal = prediction_raw[values.argmax()]
    return optimal.cpu().numpy()


@torch.no_grad()
def propose_plan_continuous(model, x):
    latent = torch.zeros([1, model.trajectory_embd], device="cuda")
    prediction = model.decode(latent, x[:, 0, :model.observation_dim])
    prediction = prediction.reshape([-1, model.transition_dim])
    return prediction.cpu().numpy()