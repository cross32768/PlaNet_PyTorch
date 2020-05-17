import gym
import numpy as np
from viewer import OpenCVImageViewer


class GymWrapper(object):
    """
    Gym interface wrapper for dm_control env wrapped by pixels.Wrapper
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env):
        self._env = env
        self._viewer = None

    def __getattr(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        return gym.spaces.Box(0, 255, obs_spec['pixels'].shape, dtype=np.uint8)

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        return gym.spaces.Box(action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['pixels']
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': time_step.discount}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation['pixels']
        return obs

    def render(self, mode='human', **kwargs):
        if not kwargs:
            kwargs = self._env._render_kwargs

        img = self._env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self._viewer is None:
                self._viewer = OpenCVImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen
        else:
            raise NotImplementedError


class RepeatAction(gym.Wrapper):
    """
    Action repeat wrapper to act same action repeatedly
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
