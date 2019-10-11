﻿import argparse
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
    from utils.utils import TabularPolicy, TabularValueFun
    from algos.tabular_value_iteration import ValueIteration
    from envs import ASRSEnv, ExpandStateWrapper

    assert np.array(eval(args.storage_shape)).prod() == len(eval(args.dist_param)), 'storage_shape should be consistent with dist_param length'
    env = ExpandStateWrapper(ASRSEnv(eval(args.storage_shape), dist_param = eval(args.dist_param)))

    env_name = env.__name__
    exp_dir = os.getcwd() + '/data/version1/%s/policy_type%s_temperature%s/' % (env_name, args.policy_type, args.temperature)
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], level=eval(args.logger_level))
    args_dict = vars(args)
    args_dict['env'] = env_name
    json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

    policy = TabularPolicy(env)
    value_fun = TabularValueFun(env)
    algo = ValueIteration(env,
                          value_fun,
                          policy,
                          policy_type=args.policy_type,
                          render=render,
                          temperature=args.temperature)
    algo.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", "-p", type=str, default='deterministic', choices=["deterministic", "max_ent"],
                        help="Whether to train a deterministic policy or a maximum entropy one")
    parser.add_argument("--render", "-r", action='store_true', help="Vizualize the policy and contours when training")
    parser.add_argument("--temperature", "-t", type=float, default=1.,
                        help="Temperature parameter for maximum entropy policies")
    parser.add_argument("--logger_level", "-l", type=str, default='INFO', choices=["DEBUG", "INFO", "WARN", "ERROR", "DISABLED"],
                        help="Level of the logger to be print")
    parser.add_argument("--storage_shape", "-s", type=str, default='(2, 1)',
                        help="ASRSEnv storage shape")
    parser.add_argument("--dist_param", "-prob", type=str, default='[0.1, 0.9]',
                        help="ASRSEnv dist_param, the order probability")
    args = parser.parse_args()
    main(args)
