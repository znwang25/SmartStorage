from envs.asrs_env import ASRSEnv
from utils.utils import RandomPolicy
from utils.plot import rollout
import moviepy.editor as mpy

videos = []
contours = []
returns = []
fig = None
itr = 0
env = ASRSEnv((2,3),dist_param = [0.01, 0.2, 0.4, 0.5, 0.7, 0.9])
policy = RandomPolicy(env)

average_return, video = rollout(env, policy, render=True,
                                num_rollouts=1)
videos += video

fps = int(4/getattr(env, 'dt', 0.1))
if videos:
    clip = mpy.ImageSequenceClip(videos, fps=fps)
    clip.write_videofile('data/roll_outs.mp4')
env.close()
