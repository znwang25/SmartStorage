﻿import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import logger
from gym import spaces
import time


def plot_returns(returns,varname='Return'):
    plt.close()
    plt.clf()
    plt.plot(range(len(returns)), returns)
    plt.xlabel("Iterations")
    plt.ylabel("Average %s" % varname)
    plt.savefig('%s/learning_curve_%s.png' % (logger.get_dir(),varname))


def plot_contour(env, value_fun, save=False, fig=None, iteration=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    ax = fig.axes[0]
    # if not hasattr(env, '_wrapped_env'):
    #     if isinstance(env.observation_space, spaces.Discrete):
    #         if env.obs_dim == 2:
    #             V = value_fun.get_values()[:-1].reshape(env._size, env._size)
    #         elif env.obs_dim == 1:
    #             V = value_fun.get_values()[:-1].reshape(env._size, 1)
    #         else:
    #             return None, fig
    #     else:
    #         low, high = env.observation_space.low, env.observation_space.high
    #         if low.shape[0] == 2:
    #             points = np.array(np.meshgrid(*[np.linspace(l, h, 41) for l, h in zip(low, high)])).reshape(2, -1).T
    #             V = value_fun.get_values(points).reshape(41, 41)[: ,::-1]
    #         elif low.shape[0] == 1:
    #             points = np.stack([np.linspace(l, h, 41) for l, h in zip(low, high)])
    #             V = value_fun.get_values(points.reshape(-1, 2))
    #         else:
    #             return None, fig

    #     V = ((V - V.min())/(V.max() - V.min() + 1e-6)).T[::-1, :]
    #     image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.).astype(np.uint8)
    #     if env.__class__.__name__ == 'GridWorldEnv':
    #         image[env._grid.astype(bool),:] = 0
    #     ax.imshow(image)
    # elif env.obs_dim == 1:
    if env.obs_dim == 1:
        V = np.expand_dims(value_fun.get_values(), 0)
        V = (V - V.min())/(V.max() - V.min())
        ax.imshow(V, vmin=0, vmax=1, cmap=plt.cm.coolwarm, origin='lower')
    elif env.obs_dim == 2:
        bx, by = env.storage_shape
        V = value_fun.get_values().reshape(bx, by)
        V = (V - V.min())/(V.max() - V.min() + 1e-6)
        ax.imshow(V, vmin=0, vmax=1, cmap=plt.cm.coolwarm, origin='lower')
    else:
        return None, fig

    if iteration is not None:
        ax.set_title('Iteration %d' % iteration)
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    if save:
        fig.savefig('%s/contour.png' % logger.get_dir())

    return image, fig


def rollout(env, policy, num_rollouts=1, max_path_length=10, last_max_path_length=600, render=True, iteration=None, last_iteration=False):
    R = 0.
    delay_c = 0.
    images = []
    if last_iteration:
        max_path_length = last_max_path_length
        print("This is last rollout!")
    if num_rollouts > 1:
        obs = env.vec_reset(num_rollouts)
        # print(f'long term p: {env.long_term_2p/2}')
        for t in range(max_path_length):
            if last_iteration and t%20==1:
                print('\r Percentage finished: {}%'.format(round(t/last_max_path_length*100),1), end="")
            a = policy.get_action(obs)
            if render:
                img = env.render('rgb_array', iteration)
            else:
                img = None
            obs, reward, delay_cost = env.vec_rollout_step(a)
            # print(f'Dynamic p: {env.dist_param}')
            R += reward
            delay_c += delay_cost
            images.append(img)
        print('\n')
    else:
        obs = env.reset()
        for t in range(max_path_length):
            a = policy.get_action(obs)
            if render:
                img = env.render('rgb_array', iteration)
            else:
                img = None
            obs, reward, delay_cost = env.step(a)
            print(f'Dynamic p: {env.dist_param}')
            R += reward
            delay_c += delay_cost
            images.append(img)
    return np.sum(R)/num_rollouts, np.sum(delay_c)/num_rollouts, images
