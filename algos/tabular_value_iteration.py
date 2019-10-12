import matplotlib.pyplot as plt
from utils.plot import plot_contour, rollout, plot_returns
import logger
import numpy as np
import moviepy.editor as mpy
import time
from utils.utils import SparseArray



class ValueIteration(object):
    """
    Tabular Value Iteration algorithm.

    -- UTILS VARIABLES FOR RUNNING THE CODE -- (feel free to play with them but we do not ask it for the homework)
        * policy (TabularPolicy):

        * precision (float): tolerance for the final values (determines the amount of iterations)

        * policy_type (str): whether the policy is deterministic or max-ent

        * log_itr (int): number of iterations between logging

        * max_itr (int): maximum number of iterations

        * render (bool): whether to render or not

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

        * self.transitions (np.ndarray): transition matrix of size (S, A, S)

        * self.rewards (np.ndarray): reward matrix of size (S, A, S)

        * self.discount (float): discount factor of the problem

        * self.temperature (float): temperature for the maximum entropy policies


    """
    def __init__(self,
                 env,
                 value_fun,
                 policy,
                 precision=1e-3,
                 log_itr=1,
                 render_itr=2,
                 policy_type='deterministic',
                 max_itr=80,
                 render=False,
                 num_rollouts=1,
                 max_path_length=10,
                 temperature=1.,
                 ):
        self.env = env
        self.transitions = env.transitions
        self.rewards = env.rewards
        self.value_fun = value_fun
        self.policy = policy
        self.discount = env.discount
        self.precision = precision
        self.log_itr = log_itr
        assert policy_type in ['deterministic', 'max_ent']
        self.policy_type = policy_type
        self.max_itr = max_itr
        self.render_itr = render_itr
        self.render = render
        self.num_rollouts = num_rollouts
        self.max_path_length = max_path_length
        self.temperature = temperature
        self.eps = 1e-8


    def train(self):
        next_v = 1e6
        v = self.value_fun.get_values()
        itr = 0
        videos = []
        contours = []
        returns = []
        fig = None

        while not self._stop_condition(itr, next_v, v) and itr < self.max_itr:
            log = itr % self.log_itr == 0
            render = (itr % self.render_itr == 0) and self.render
            if log:
                next_pi = self.get_next_policy()
                self.policy.update(next_pi)
                average_return, video = rollout(self.env, self.policy, render=render,
                                                num_rollouts=self.num_rollouts, max_path_length=self.max_path_length,iteration=itr)
                if render:
                    contour, fig = plot_contour(self.env, self.value_fun, fig=fig, iteration=itr)
                    contours += [contour] * len(video)
                    videos += video
                returns.append(average_return)
                logger.logkv('Iteration', itr)
                logger.logkv('Average Returns', average_return)
                logger.dumpkvs()
            next_v = self.get_next_values()
            self.value_fun.update(next_v)
            itr += 1

        next_pi = self.get_next_policy()
        print(next_pi)
        print(self.value_fun.get_values())
        self.policy.update(next_pi)
        contour, fig = plot_contour(self.env, self.value_fun, save=True, fig=fig, iteration=itr)
        average_return, video = rollout(self.env, self.policy,
                                        render=self.render, num_rollouts=self.num_rollouts, max_path_length=self.max_path_length, iteration=itr)
        plot_returns(returns)
        if self.render:
            videos += video
            contours += [contour]
        logger.logkv('Iteration', itr)
        logger.logkv('Average Returns', average_return)

        fps = int(4/getattr(self.env, 'dt', 0.1))
        if contours and contours[0] is not None:
            clip = mpy.ImageSequenceClip(contours, fps=fps)
            clip.write_videofile('%s/contours_progress.mp4' % logger.get_dir())

        if videos:
            clip = mpy.ImageSequenceClip(videos, fps=fps)
            clip.write_videofile('%s/roll_outs.mp4' % logger.get_dir())

        plt.close()

    # def get_next_values(self):
    #     """
    #     Next values given by the Bellman equation

    #     :return np.ndarray with the values for each state, shape (num_states,)
    #     For the maximum entropy policy, to compute the unnormalized probabilities make sure:
    #                                 1) Before computing the exponientated value substract the maximum value per state
    #                                    over all the actions.
    #                                 2) Add self.eps to them
    #     """

    #     """ INSERT YOUR CODE HERE"""
    #     v = self.value_fun.get_values()
    #     n_state = self.env.num_states
    #     n_action = self.env.num_actions
    #     Q_value = np.zeros((n_state, n_action))
    #     for s in range(n_state):
    #         for a in range(n_action):
    #             if isinstance(self.transitions, SparseArray):
    #                 non_zero_next_states = self.transitions._idxs[s,a]
    #                 Q_value[s,a] = (self.transitions[s,a]*(self.rewards[s,a]+self.discount*v[non_zero_next_states])).sum()
    #             else:
    #                 Q_value[s,a] = (self.transitions[s,a] * (self.rewards[s,a]+self.discount*v)).sum() 
    #     if self.policy_type == 'deterministic':
    #         next_v = Q_value.max(axis=1)
    #         self._next_policy = Q_value.argmax(axis=1)
    #     elif self.policy_type == 'max_ent':
    #         next_v = np.log(np.exp(1/self.temperature*(Q_value-Q_value.max(axis=1).reshape(-1, 1))).sum(axis=1))*self.temperature+Q_value.max(axis=1)
    #         self._next_policy = np.exp(Q_value-Q_value.max(axis=1).reshape(-1, 1))+self.eps
    #         self._next_policy /= np.linalg.norm(self._next_policy, ord=1, axis=1, keepdims=True)
    #     else:
    #         raise NotImplementedError
    #     return next_v

    def get_next_values(self):
            """
            Next values given by the Bellman equation

            :return np.ndarray with the values for each state, shape (num_states,)
            For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                        1) Before computing the exponientated value substract the maximum value per state
                                        over all the actions.
                                        2) Add self.eps to them
            """

            """ INSERT YOUR CODE HERE"""
            if self.policy_type == 'deterministic':
                v = self.value_fun.get_values()
                next_v = (self.transitions * (self.rewards + self.discount * v)).sum(axis=2).max(axis=1)
            elif self.policy_type == 'max_ent':
                v = self.value_fun.get_values()
                Q = (self.transitions * (self.rewards + self.discount * v)).sum(axis=2)
                Q_normalized = Q - np.multiply(Q.max(axis=1).reshape([-1, 1]), np.ones(Q.shape[1]))
                next_v = self.temperature * np.log(np.exp(Q_normalized/self.temperature).sum(axis=1)) + Q.max(axis=1)
                """ Your code ends here"""
            else:
                raise NotImplementedError
            return next_v

    def get_next_policy(self):
        """
        Next policy probabilities given by the Bellman equation

        :return np.ndarray with the policy probabilities for each state and actions, shape (num_states, num_actions)
        For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                    1) Before computing the exponientated value substract the maximum value per state
                                       over all the actions.
                                    2) Add self.eps to them
        """

        """INSERT YOUR CODE HERE"""
        if self.policy_type == 'deterministic':
            v = self.value_fun.get_values()
            Q = (self.transitions * (self.rewards + self.discount * v)).sum(axis=2)
            optimal_action_first_index = np.argmax(Q, axis=1)
            pi = np.zeros(Q.shape)
            for i in range(pi.shape[0]):
                pi[i, optimal_action_first_index[i]] = 1.0
        elif self.policy_type == 'max_ent':
            v = self.value_fun.get_values()
            Q = (self.transitions * (self.rewards + self.discount * v)).sum(axis=2)
            Q_normalized = Q - np.multiply(Q.max(axis=1).reshape([-1, 1]), np.ones(Q.shape[1]))
            proportion = np.exp(Q_normalized) + self.eps
            pi = proportion / proportion.sum(axis=1).reshape([-1, 1])
            """ Your code ends here"""
        else:
            raise NotImplementedError
        return pi


    # def get_next_policy(self):
    #     """
    #     Next policy probabilities given by the Bellman equation

    #     :return np.ndarray with the policy probabilities for each state and actions, shape (num_states, num_actions)
    #     For the maximum entropy policy, to compute the unnormalized probabilities make sure:
    #                                 1) Before computing the exponientated value substract the maximum value per state
    #                                    over all the actions.
    #                                 2) Add self.eps to them
    #     """

    #     """INSERT YOUR CODE HERE"""
    #     if self.policy_type == 'deterministic':
    #         if not hasattr(self,'_next_policy'):
    #             self.get_next_values()
    #         pi = self._next_policy
    #     elif self.policy_type == 'max_ent':
    #         if not hasattr(self,'_next_policy'):
    #             self.get_next_values()
    #         pi = self._next_policy
    #     else:
    #         raise NotImplementedError
    #     return pi

    def _stop_condition(self, itr, next_v, v):
        rmax = np.max(np.abs(self.env.rewards))
        cond1 = np.max(np.abs(next_v - v)) < self.precision/(2 * self.discount/(1 - self.discount))
        cond2 = self.discount ** itr * rmax/(1 - self.discount) < self.precision
        return cond1 or cond2
