import argparse
import json
import os
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, RewardModel
from wrappers import GymWrapper, RepeatAction


def main():
    parser = argparse.ArgumentParser(description='Test learned model')
    parser.add_argument('dir', type=str, help='log directory to load learned model')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('-H', '--horizon', type=int, default=12)
    parser.add_argument('-I', '--N-iterations', type=int, default=10)
    parser.add_argument('-J', '--N-candidates', type=int, default=1000)
    parser.add_argument('-K', '--N-top-candidates', type=int, default=100)
    args = parser.parse_args()

    # define environment and apply wrapper
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define models
    with open(os.path.join(args.dir, 'args.json'), 'r') as f:
        train_args = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(train_args['state_dim'], env.action_space.shape[0],
                                    train_args['rnn_hidden_dim'], train_args['hidden_dim'],
                                    train_args['min_stddev']).to(device)
    reward_model = RewardModel(train_args['state_dim'],
                               train_args['rnn_hidden_dim'],
                               train_args['hidden_dim']).to(device)

    # load learned parameters
    encoder.load_state_dict(torch.load(os.path.join(args.dir, 'encoder.pth')))
    rssm.load_state_dict(torch.load(os.path.join(args.dir, 'rssm.pth')))
    reward_model.load_state_dict(torch.load(os.path.join(args.dir, 'reward_model.pth')))

    # define agent
    cem_agent = CEMAgent(encoder, rssm, reward_model,
                         args.horizon, args.N_iterations,
                         args.N_candidates, args.N_top_candidates)

    # test learnged model in the environment
    for episode in range(args.episodes):
        cem_agent.reset()
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = cem_agent(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if args.render:
                env.render(height=256, width=256, camera_id=0)

        print('Total test reward at episode [%4d/%4d] is %f' %
              (episode+1, args.episodes, total_reward))


if __name__ == '__main__':
    main()
