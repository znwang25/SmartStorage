﻿from gym import spaces
from utils.plot import rollout, plot_returns, plot_contour
# from utils.utils import upsample
import logger
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt


class FunctionApproximateValueIteration(object):
    """
    Value Iteration for continous state environments.

    -- UTILS VARIABLES FOR RUNNING THE CODE --
    * self.get_states_and_transitions(): random states, its subsequent next states, rewards, dones; for the specified
        number of actions and batch size

    * policy (LookAheadPolicy)

    * batch_size (int): number of states to sample per iteration

    * num_acts (int): number of actions to sample to compute the max over the value function. If the env is discrete and
                      and num_acts is None it will do the max over all the actions

    * learning_rate: learning rate of the gradient descent algorithm

    * max_iter (int): maximum number of iterations


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
     * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

    * self.discount (float): discount factor of the problem

    * self.num_acts (int): number of actions used to maximize the value function.
    to the target values.
    """
    def __init__(self,
                 env,
                 value_fun,
                 policy,
                 batch_size,
                 num_acts,
                 learning_rate=0.1,
                 max_itr=2500,
                 log_itr=2,
                 render_itr=25,
                 num_rollouts=10,
                 render=True,
                 ):
        self.env = env
        self.discount = env.discount
        self.value_fun = value_fun
        self.policy = policy
        self.max_itr = max_itr
        self.log_itr = log_itr
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size
        self.num_acts = self.env.action_space.n if isinstance(env.action_space, spaces.Discrete) else num_acts
        self.num_acts = self.env.num_actions
        self.render_itr = render_itr
        self.render = render

    def train(self):
        # params = self.value_fun._params
        videos = []
        contours = []
        returns = []
        delay_cs = []
        fig = None
        for itr in range(self.max_itr):
            self.value_fun_update()

            log = itr % self.log_itr == 0 or itr == self.max_itr - 1
            render = (itr % self.render_itr == 0) and self.render
            if log:
                average_return, avg_delay_cost, video = rollout(self.env, self.policy, num_rollouts=self.num_rollouts, render=render, iteration=itr)
                if render:
                    # contour, fig = plot_contour(self.env, self.value_fun, fig=fig, iteration=itr)
                    # contours += [contour]
                    videos += video
                returns.append(average_return)
                delay_cs.append(avg_delay_cost)
                logger.logkv('Iteration', itr)
                logger.logkv('Average Returns', average_return)
                logger.logkv('Average Delayed Costs', avg_delay_cost)
                logger.dumpkvs()

        plot_returns(returns)
        plot_returns(delay_cs,'delayed_cost')
        # plot_contour(self.env, self.value_fun, save=True, fig=fig)

        # if contours and contours[0] is not None:
        #     contours = list(upsample(np.array(contours), 10))
        #     clip = mpy.ImageSequenceClip(contours, fps=10)
        #     clip.write_videofile('%s/contours_progress.mp4' % logger.get_dir())

        if videos:
            fps = int(4 / getattr(self.env, 'dt', 0.1))
            clip = mpy.ImageSequenceClip(videos, fps=fps)
            clip.write_videofile('%s/learning_progress.mp4' % logger.get_dir())

        plt.close()

    def value_fun_update(self):
        """
        L2 Loss of the bellman error w.r.t to parametres of our value function
        :param params:
        :return: loss function
        """
        states, next_states, rewards = self.get_states_and_transitions()
        num_acts, num_states = self.num_acts, self.batch_size
        Q = rewards + self.discount * self.value_fun.get_values(next_states)
        V_bar = Q.reshape([num_acts, num_states]).max(axis=0)

        states_distinct = states[:num_states]
        self.value_fun.update(states_distinct,V_bar)

    def get_states_and_transitions(self):
        num_acts, num_states = self.num_acts, self.batch_size
        states = self.env.sample_states(num_states)
        actions = self.env.sample_actions(num_acts, all=True)

        states = np.tile(states.T, num_acts).T
        actions = np.repeat(actions, num_states, axis=0)
        self.env.vec_set_state(states)
        next_states, rewards, delay_costs = self.env.vec_step(actions)
        return states, next_states, rewards