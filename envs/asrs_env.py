import math
import gym
import matplotlib
from gym import spaces
from gym.utils import seeding
import numpy as np

class ASRSEnv(gym.Env):

    """
    Description:
    There is a storage warehouse with M= W * H bins. M (or W * H) types of products will be store in this warehouse. Each period, there will be an array of orders coming in. 
    A robot can exchange the positions of two bins. The goal is to find the optimal storage plan to make best fullfil the orders.

    Observation:
        Type: 
        Num	Observation               
        0	Current Storage Map    np.array()   (M,)     
        1	Current Period Order   np.array()    (M,)

    Actions:
        Type:
        Num	of Action: M Choose 2
        e.g  (a , b) switch bin a with bin b

    Reward:
        Reward is -1 for every step taken

    Starting State:
        A random permutation.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, num_products,dist_param):
        self.seed()
        self.num_products = num_products
        self.dist_param = np.array(dist_param)
        self.distribution = np.random.exponential
        self.time_to_next = np.zeros(num_products)
        self.viewer = None
        storage_map = np.random.permutation(num_products)
        order = np.zeros(num_products)
        self.state = np.concatenate([storage_map,order])
        self._states = None
        self._num_envs = None

        self.steps_beyond_done = None
        self.vectorized = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action=None):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        storage_map = self.state[:self.num_products]

        arrived = self.time_to_next<1
        order = arrived
        self.time_to_next = self.time_to_next-1
        if arrived.any():
            self.time_to_next[arrived] = self.distribution(self.dist_param[arrived])
        if action is not None:
            storage_map[action] = storage_map[[action[1], action[0]]]
        self.state = np.concatenate([storage_map,order])
        reward = -1.0
        return np.array(self.state).copy(), reward

    def vec_step(self, actions):
        # assert list(map(self.action_space.contains, actions)).all()
        assert self._states is not None
        state = self._states
        x, x_dot, theta, theta_dot = state.T
        force = (2 * actions - 1) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self._states = np.stack([x, x_dot, theta, theta_dot], axis=-1)
        dones = (x < -self.x_threshold) \
               + (x > self.x_threshold) \
               + (theta < -self.theta_threshold_radians) \
               + (theta > self.theta_threshold_radians) \
               + (x_dot > self.vel_threshold) \
               + (x_dot < -self.vel_threshold) \
               + (theta_dot > self.vel_threshold) \
               + (theta_dot < -self.vel_threshold)

        dones = dones.astype(bool)
        rewards = np.ones((self._num_envs,))
        """
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        """

        return np.array(self._states).copy(), rewards, dones, {}

    def vec_reset(self, num_envs=None):
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._states = self.np_random.uniform(low=-0.05, high=0.05, size=(num_envs, 4))
        return np.array(self._states).copy()

    def vec_set_state(self, states):
        self._num_envs = len(states)
        self._states = states.copy()

    def set_state(self, state):
        self.state = state

    def reset(self):
        self._states = None
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        storage_map = np.random.permutation(num_products)
        order = np.zeros(num_products)
        self.state = np.concatenate([storage_map,order])
        self.time_to_next = np.zeros(num_products)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human', iteration=None):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
