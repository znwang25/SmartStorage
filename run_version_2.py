import argparse
import os
import logger
import json
import numpy as np
from logger import DEBUG, INFO, WARN, ERROR, DISABLED 

def main(args):
    render = args.render
    if not render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from utils.utils import TabularPolicy, LookAheadPolicy, SimpleMaxPolicy
    from utils.value_function import CNNValueFun, TabularValueFun
    from algos.function_approximate_value_iteration import FunctionApproximateValueIteration
    from envs import ASRSEnv, MapAsPicEnv

    assert np.array(eval(args.storage_shape)).prod() == len(eval(args.dist_param)), 'storage_shape should be consistent with dist_param length'
    env = MapAsPicEnv(ASRSEnv(eval(args.storage_shape), origin_coord=eval(args.exit_coord), dist_param = eval(args.dist_param)))

    env_name = env.__name__
    exp_dir = os.getcwd() + '/data/version2/%s/policy_type%s_envsize_%s/' % (env_name, args.policy_type,np.array(eval(args.storage_shape)).prod())
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], level=eval(args.logger_level))
    args_dict = vars(args)
    args_dict['env'] = env_name
    json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

    value_fun = CNNValueFun(env)
    policy = SimpleMaxPolicy(env,
                            value_fun,
                            num_acts = args.num_acts)
    # policy = LookAheadPolicy(env,
    #                         value_fun,
    #                         horizon=args.horizon,
    #                         look_ahead_type=args.policy_type,
    #                         num_acts=args.num_acts)
    algo = FunctionApproximateValueIteration(env,
                            value_fun,
                            policy,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            num_acts=args.num_acts,
                            render=render,
                            num_rollouts = args.num_rollouts,
                            max_itr=args.max_iter,
                            log_itr=5)
    algo.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action='store_true', help="Vizualize the policy and contours when training")
    parser.add_argument("--policy_type", "-p", type=str, default='rs', choices=['cem', 'rs'],
                        help='Type of policy to use. Whether to use look ahead with cross-entropy \
                        method or random shooting')
    parser.add_argument("--horizon", "-H", type=int, default=1,
                        help='Planning horizon for the look ahead policy')
    parser.add_argument("--max_iter", "-i", type=int, default=250,
                        help='Maximum number of iterations for the value iteration algorithm')
    parser.add_argument("--logger_level", "-l", type=str, default='INFO', choices=["DEBUG", "INFO", "WARN", "ERROR", "DISABLED"],
                        help="Level of the logger to be print")
    parser.add_argument("--storage_shape", "-s", type=str, default='(2, 1)',
                        help="ASRSEnv storage shape")
    parser.add_argument("--exit_coord", "-e", type=str, default='None',
                        help="Coordinate of the exit, default is the origin")
    parser.add_argument("--dist_param", "-prob", type=str, default='[0.1, 0.9]',
                        help="ASRSEnv dist_param, the order probability")
    parser.add_argument("--num_rollouts", "-nr", type=int, default=1,
                        help="Number of rollouts used to evaluate policy")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help='Learning rate for training the value function')
    parser.add_argument("--batch_size", "-bs", type=int, default=256,
                        help='batch size for training the value function')
    parser.add_argument("--num_acts", "-a", type=int, default=10,
                        help='Number of actions sampled for maximizing the value function')
    args = parser.parse_args()
    main(args)
