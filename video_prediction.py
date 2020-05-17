import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import preprocess_obs
from wrappers import GymWrapper, RepeatAction


def save_video_as_gif(frames):
    """
    make video with given frames and save as "video_prediction.gif"
    """
    plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        plt.title('Left: GT frame' + ' '*20 + 'Right: predicted frame \n Step %d' % (i))

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=150)
    anim.save('video_prediction.gif', writer='imagemagick')


def main():
    parser = argparse.ArgumentParser(description='Open-loop video prediction with learned model')
    parser.add_argument('dir', type=str, help='log directory to load learned model')
    parser.add_argument('--length', type=int, default=50,
                        help='the length of video prediction')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
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
    obs_model = ObservationModel(train_args['state_dim'],
                                 train_args['rnn_hidden_dim']).to(device)
    reward_model = RewardModel(train_args['state_dim'],
                               train_args['rnn_hidden_dim'],
                               train_args['hidden_dim']).to(device)

    # load learned parameters
    encoder.load_state_dict(torch.load(os.path.join(args.dir, 'encoder.pth')))
    rssm.load_state_dict(torch.load(os.path.join(args.dir, 'rssm.pth')))
    obs_model.load_state_dict(torch.load(os.path.join(args.dir, 'obs_model.pth')))
    reward_model.load_state_dict(torch.load(os.path.join(args.dir, 'reward_model.pth')))

    # define agent
    cem_agent = CEMAgent(encoder, rssm, reward_model,
                         args.horizon, args.N_iterations,
                         args.N_candidates, args.N_top_candidates)

    # open-loop video prediction
    # select starting point of open-loop prediction randomly
    starting_point = torch.randint(1000 // args.action_repeat - args.length, (1,)).item()
    # interact in environment until starting point and charge context in cem_agent.rnn_hidden
    obs = env.reset()
    for _ in range(starting_point):
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)

    # preprocess observatin and embed by encoder
    preprocessed_obs = preprocess_obs(obs)
    preprocessed_obs = torch.as_tensor(preprocessed_obs, device=device)
    preprocessed_obs = preprocessed_obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        embedded_obs = encoder(preprocessed_obs)

    # compute state using embedded observation
    # NOTE: after this, state is updated only using prior,
    #       it means model doesn't see observation
    rnn_hidden = cem_agent.rnn_hidden
    state = rssm.posterior(rnn_hidden, embedded_obs).sample()
    frame = np.zeros((64, 128, 3))
    frames = []
    for _ in range(args.length):
        # action is selected same as training time (closed-loop)
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)

        # update state and reconstruct observation with same action
        action = torch.as_tensor(action, device=device).unsqueeze(0)
        with torch.no_grad():
            state_prior, rnn_hidden = rssm.prior(state, action, rnn_hidden)
            state = state_prior.sample()
            predicted_obs = obs_model(state, rnn_hidden)

        # arrange GT frame and predicted frame in parallel
        frame[:, :64, :] = preprocess_obs(obs)
        frame[:, 64:, :] = predicted_obs.squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
        frames.append((frame + 0.5).clip(0.0, 1.0))

    save_video_as_gif(frames)

if __name__ == '__main__':
    main()
