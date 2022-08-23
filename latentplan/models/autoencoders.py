from latentplan.models.transformers import *
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, condition_size):

        super().__init__()
        layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, condition_size):
        super().__init__()
        self.MLP = nn.Sequential()

        input_size = latent_size + condition_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z):
        x = self.MLP(z)

        return x

class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim

        encoder_layer_sizes = [self.trajectory_input_length, 512, 256]
        decoder_layer_sizes = [256, 512, self.trajectory_input_length]

        self.encoder = Encoder(
            encoder_layer_sizes, config.trajectory_embd, 0)
        self.decoder = Decoder(
            decoder_layer_sizes, config.trajectory_embd, self.condition_size)

    def encode(self, X):
        """
            X: [B x T x transition_dimension]
        """
        B, _, _ = X.shape
        inputs = torch.reshape(X, shape=[B, -1])
        latents = self.encoder(inputs)
        return latents

    def decode(self, latents, state):
        """
            latents: [B x latent_size]
            state: [B x observation_dimension]
        """
        B, _ = latents.shape
        state_flat = torch.reshape(state, shape=[B, -1])
        inputs = torch.cat([state_flat, latents], dim=-1)
        reconstructed = self.decoder(inputs)
        return reconstructed

class SymbolWiseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_size = config.trajectory_embd
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = 4*(config.block_size//config.transition_dim-1)
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim

        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))

        self.state_emb = nn.Linear(self.observation_dim, self.embedding_dim)
        self.action_emb = nn.Linear(self.action_dim, self.embedding_dim)
        self.reward_emb = nn.Linear(1, self.embedding_dim)
        self.value_emb = nn.Linear(1, self.embedding_dim)

        self.pred_state = nn.Linear(self.embedding_dim, self.observation_dim)
        self.pred_action = nn.Sequential(nn.Linear(self.embedding_dim, self.action_dim))
        self.pred_reward = nn.Linear(self.embedding_dim, 1)
        self.pred_value = nn.Linear(self.embedding_dim, 1)

        self.linear_means = nn.Linear(self.embedding_dim, self.latent_size)
        self.linear_log_var = nn.Linear(self.embedding_dim, self.latent_size)
        self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, self.embedding_dim)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)


    def encode(self, joined_inputs):
        b, t, joined_dimension = joined_inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        states = joined_inputs[:, :, :self.observation_dim]
        actions = joined_inputs[:, :, self.observation_dim:self.observation_dim + self.action_dim]
        rewards = joined_inputs[:, :, -2, None]
        values = joined_inputs[:, :, -1, None]

        state_embeddings = self.state_emb(states)
        action_embeddings = self.action_emb(actions)
        reward_embeddings = self.reward_emb(rewards)
        value_embeddings = self.value_emb(values)

        token_embeddings = torch.stack([state_embeddings, action_embeddings, reward_embeddings, value_embeddings],
                                       dim=1) \
            .permute([0, 2, 1, 3]).reshape(b, 4 * t, self.embedding_dim)
        ## [ 1 x 4T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :4 * t, :]  # each position maps to a (learnable) vector
        ## [ B x 4T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.encoder(x)
        ## [ B x 4T x embedding_dim ]
        trajectory_feature = x.max(dim=1).values
        means = self.linear_means(trajectory_feature)
        log_vars = self.linear_log_var(trajectory_feature)
        return means, log_vars

    def decode(self, latents, state):
        """
            latents: [B x latent_size]
            state: [B x observation_dimension]
        """
        B, _ = latents.shape
        state_flat = torch.reshape(state, shape=[B, -1])
        inputs = torch.cat([state_flat, latents], dim=-1)
        inputs = self.latent_mixing(inputs)
        inputs = inputs[:, None, :] + self.pos_emb[:, :]
        x = self.decoder(inputs)
        x = self.ln_f(x)

        x = x.reshape(B, -1, 4, self.embedding_dim).permute([0,2,1,3])

        ## [B x T x obs_dim]
        state_pred = self.pred_state(x[:,1]) # next state
        action_pred = self.pred_action(x[:,0]) # current action
        reward_pred = self.pred_reward(x[:,1]) # current reward
        value_pred = self.pred_value(x[:,1]) # current value
        joined_pred = torch.cat([state_pred, action_pred, reward_pred, value_pred], dim=-1)

        return joined_pred


class StepWiseTranformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_size = config.trajectory_embd
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim

        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))

        self.embed = nn.Linear(self.transition_dim, self.embedding_dim)

        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)

        self.linear_means = nn.Linear(self.embedding_dim, self.latent_size)
        self.linear_log_var = nn.Linear(self.embedding_dim, self.latent_size)
        self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, self.embedding_dim)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)


    def encode(self, joined_inputs):
        b, t, joined_dimension = joined_inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.embed(joined_inputs)

        ## [ 1 x 4T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        ## [ B x 4T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.encoder(x)
        ## [ B x 4T x embedding_dim ]
        trajectory_feature = x.max(dim=1).values
        means = self.linear_means(trajectory_feature)
        log_vars = self.linear_log_var(trajectory_feature)
        return means, log_vars

    def decode(self, latents, state):
        """
            latents: [B x latent_size]
            state: [B x observation_dimension]
        """
        B, _ = latents.shape
        state_flat = torch.reshape(state, shape=[B, -1])
        inputs = torch.cat([state_flat, latents], dim=-1)
        inputs = self.latent_mixing(inputs)
        inputs = inputs[:, None, :] + self.pos_emb[:, :]
        x = self.decoder(inputs)
        x = self.ln_f(x)

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        return joined_pred