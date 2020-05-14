import torch
from torch.distributions import Normal
from utils import preprocess_obs


class MPCAgent:
    def __init__(self, encoder, rssm, reward_model,
                 horizon, N_iterations, N_candidates, N_top_candidates):
        self.encoder = encoder
        self.rssm = rssm
        self.reward_model = reward_model

        self.horizon = horizon
        self.N_iterations = N_iterations
        self.N_candidates = N_candidates
        self.N_top_candidates = N_top_candidates

        self.device = next(self.reward_model.parameters()).device
        self.prev_rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim).to(self.device)

    def __call__(self, obs):
        obs = preprocess_obs(obs)
        obs = torch.FloatTensor(obs).transpose(1, 2).transpose(0, 1).unsqueeze(0)
        obs = obs.to(self.device)
        with torch.no_grad():
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.posterior(self.prev_rnn_hidden, embedded_obs)
            action_dist = Normal(torch.zeros(self.horizon, self.rssm.action_dim),
                                 torch.ones(self.horizon, self.rssm.action_dim))
            for itr in range(self.N_iterations):
                action_candidates = \
                    action_dist.sample([self.N_candidates]).transpose(0, 1).to(self.device)
                total_predicted_reward = torch.zeros(self.N_candidates)
                state = state_posterior.sample([self.N_candidates]).squeeze()
                rnn_hidden = self.prev_rnn_hidden.repeat([self.N_candidates, 1])
                for t in range(self.horizon):
                    next_state_prior, rnn_hidden = \
                        self.rssm.prior(state, action_candidates[t], rnn_hidden)
                    state = next_state_prior.sample()
                    total_predicted_reward += self.reward_model(state, rnn_hidden).squeeze().cpu()
                top_indexes = \
                    total_predicted_reward.argsort(descending=True)[: self.N_top_candidates]
                top_action_candidates = action_candidates[:, top_indexes, :]
                mean = top_action_candidates.mean(dim=1)
                stddev = (top_action_candidates - mean.unsqueeze(1)
                          ).abs().sum(dim=1) / (self.N_top_candidates - 1)
                action_dist = Normal(mean, stddev)

        action = mean[0]
        with torch.no_grad():
            _, self.prev_rnn_hidden = \
                self.rssm.prior(state_posterior.sample(), action.unsqueeze(0), self.prev_rnn_hidden)
        return action.cpu().numpy()
