import argparse
import torch
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from dm_control import suite
from dm_control.suite.wrappers import pixels
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import ReplayBuffer, preprocess_obs
from wrappers import GymWrapper, RepeatAction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--state-dim', type=int, default=30)
    parser.add_argument('--rnn-hidden-dim', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--min-stddev', type=float, default=0.1)
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--all-episodes', type=int, default=1000)
    parser.add_argument('--seed-episodes', type=int, default=5)
    parser.add_argument('--collect-interval', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--chunk-length', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--clip-grad-norm', type=int, default=1000)
    parser.add_argument('--free-nats', type=int, default=3)
    args = parser.parse_args()

    # define env and apply wrappers
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape=env.observation_space.shape,
                                 action_dim=env.action_space.shape[0])

    # define models and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim, env.action_space.shape[0],
                                    args.rnn_hidden_dim, args.hidden_dim,
                                    args.min_stddev).to(device)
    obs_model = ObservationModel(
        args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(
        args.state_dim, args.rnn_hidden_dim, args.hidden_dim).to(device)
    all_params = (list(encoder.parameters()) +
                  list(rssm.parameters()) +
                  list(obs_model.parameters()) +
                  list(reward_model.parameters()))
    optimizer = Adam(all_params, lr=args.lr, eps=args.eps)

    # main training loop
    for episode in range(args.all_episodes):
        # collect experience
        if episode < args.seed_episodes:
            agent = env.action_space
        else:
            # TODO: use MPC Planner for this action choice
            agent = env.action_space
        obs = env.reset()
        done = False
        while not done:
            action = agent.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
        print('episode [%4d/%4d] is collected.' % (episode+1, args.all_episodes))

        # update model parameters
        for update_step in range(args.collect_interval):
            observations, actions, rewards, _ = \
                replay_buffer.sample(args.batch_size, args.chunk_length)

            # preprocess observations and transpose tensor for RNN training
            observations = preprocess_obs(observations)
            observations = torch.FloatTensor(observations).transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1).to(device)
            actions = torch.FloatTensor(actions).transpose(0, 1).to(device)
            rewards = torch.FloatTensor(rewards).transpose(0, 1).to(device)

            # embed observations with CNN
            embedded_observations = encoder(
                observations.view(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim).to(device)
            rnn_hiddens = torch.zeros(
                args.chunk_length, args.batch_size, args.rnn_hidden_dim).to(device)

            # initialize state and rnn hidden state with 0 vector
            state = torch.zeros(args.batch_size, args.state_dim).to(device)
            rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim).to(device)

            kl_loss = 0
            for l in range(args.chunk_length-1):
                next_state_prior, next_state_posterior, rnn_hidden = \
                    rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
                state = next_state_posterior.rsample()
                states[l+1] = state
                rnn_hiddens[l+1] = rnn_hidden
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                kl_loss += kl.clamp(min=args.free_nats).mean()
            kl_loss /= (args.chunk_length - 1)

            flatten_states = states.view(-1, args.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.rnn_hidden_dim)
            recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 1)

            obs_loss = mse_loss(
                recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = mse_loss(predicted_rewards[1:], rewards[:-1])

            loss = kl_loss + obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f' %
                  (update_step+1, loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item()))

if __name__ == '__main__':
    main()
