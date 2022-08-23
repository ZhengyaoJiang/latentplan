import gym
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('plotting.fullobsenv:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('plotting.fullobsenv:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('plotting.fullobsenv:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('plotting.fullobsenv:AntFullObsEnv'),
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()


class HopperFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/hopper.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos.flat[1:],
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set(self, state):
        qpos_dim = self.sim.data.qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()

class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/half_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos.flat[1:],
            self.sim.data.qpos.flat, #[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set(self, state):
        qpos_dim = self.sim.data.qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()

'''
    qpos : 15
    qvel : 14
    0-2: root x, y, z
    3-7: root quat
    7  : front L hip
    8  : front L ankle
    9  : front R hip
    10 : front R ankle
    11 : back  L hip
    12 : back  L ankle
    13 : back  R hip
    14 : back  R ankle
'''

class AntFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/ant.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class Walker2dFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/walker2d.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

registered_environments = register_environments()