import argparse
from datetime import datetime
import json
import os
from pprint import pprint
import time
import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import ReplayBuffer, preprocess_obs
from wrappers import GymWrapper, RepeatAction


def main():
    parser = argparse.ArgumentParser(description='PlaNet for DM control')
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
    parser.add_argument('--state-dim', type=int, default=30)
    parser.add_argument('--rnn-hidden-dim', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--min-stddev', type=float, default=0.1)
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--all-episodes', type=int, default=1000)
    parser.add_argument('-S', '--seed-episodes', type=int, default=5)
    parser.add_argument('-C', '--collect-interval', type=int, default=100)
    parser.add_argument('-B', '--batch-size', type=int, default=50)
    parser.add_argument('-L', '--chunk-length', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--clip-grad-norm', type=int, default=1000)
    parser.add_argument('--free-nats', type=int, default=3)
    parser.add_argument('-H', '--horizon', type=int, default=12)
    parser.add_argument('-I', '--N-iterations', type=int, default=10)
    parser.add_argument('-J', '--N-candidates', type=int, default=1000)
    parser.add_argument('-K', '--N-top-candidates', type=int, default=100)
    parser.add_argument('--action-noise-var', type=float, default=0.3)
    args = parser.parse_args()

    # Prepare logging
    log_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))
    writer = SummaryWriter(log_dir=log_dir)

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
        # collect experiences
        start = time.time()
        if episode >= args.seed_episodes:
            cem_agent = CEMAgent(encoder, rssm, reward_model,
                                 args.horizon, args.N_iterations,
                                 args.N_candidates, args.N_top_candidates)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if episode < args.seed_episodes:
                action = env.action_space.sample()
            else:
                action = cem_agent(obs)
                action += np.random.normal(0, np.sqrt(args.action_noise_var),
                                           env.action_space.shape[0])
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, args.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # update model parameters
        start = time.time()
        for update_step in range(args.collect_interval):
            observations, actions, rewards, _ = \
                replay_buffer.sample(args.batch_size, args.chunk_length)

            # preprocess observations and transpose tensor for RNN training
            observations = preprocess_obs(observations)
            observations = torch.FloatTensor(observations).to(device)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.FloatTensor(actions).to(device).transpose(0, 1)
            rewards = torch.FloatTensor(rewards).to(device).transpose(0, 1)

            # embed observations with CNN
            embedded_observations = encoder(
                observations.reshape(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            rnn_hiddens = torch.zeros(
                args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)

            # initialize state and rnn hidden state with 0 vector
            state = torch.zeros(args.batch_size, args.state_dim, device=device)
            rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

            # compute state and rnn hidden sequences and kl loss
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

            # compute reconstructed observations and predicted rewards
            flatten_states = states.view(-1, args.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.rnn_hidden_dim)
            recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 1)

            # compute loss for observation and reward
            obs_loss = mse_loss(
                recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = mse_loss(predicted_rewards[1:], rewards[:-1])

            # add all losses and update model parameters with gradient descent
            loss = kl_loss + obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f'
                  % (update_step+1,
                     loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item()))
            total_update_step = episode * args.collect_interval + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)

        print('elasped time for update: %.2fs' % (time.time() - start))

        # test to get score without exploration noise
        if (episode + 1) % args.test_interval == 0:
            start = time.time()
            cem_agent = CEMAgent(encoder, rssm, reward_model,
                                 args.horizon, args.N_iterations,
                                 args.N_candidates, args.N_top_candidates)
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = cem_agent(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, args.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

    # save learned model parameters
    torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(log_dir, 'rssm.pth'))
    torch.save(obs_model.state_dict(), os.path.join(log_dir, 'obs_model.pth'))
    torch.save(reward_model.state_dict(), os.path.join(log_dir, 'reward_model.pth'))
    writer.close()

if __name__ == '__main__':
    main()
