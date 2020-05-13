import argparse
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import ReplayBuffer
from wrappers import GymWrapper, RepeatAction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--buffer-capacity', type=int, default=100000)
    parser.add_argument('--state-dim', type=int, default=30)
    parser.add_argument('--rnn-hidden-dim', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--min-stddev', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps', type=float, default=1e-4)
    args = parser.parse_args()

    # define env
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
    
    # define models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim,
                                    env.action_space.shape[0],
                                    args.rnn_hidden_dim,
                                    args.hidden_dim,
                                    args.min_stddev).to(device)
    obs_model = ObservationModel(
        args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(
        args.state_dim, args.rnn_hidden_dim, args.hidden_dim).to(device)
    all_params = (list(encoder.parameters()) +
                  list(rssm.parameters()) +
                  list(obs_model.parameters()) +
                  list(reward_model.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps)


if __name__ == '__main__':
    main()
