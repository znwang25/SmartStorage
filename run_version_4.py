import argparse
import os
import logger
import json
import numpy as np
from logger import DEBUG, INFO, WARN, ERROR, DISABLED 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(args):
    render = args.render
    if not render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from utils.utils import TabularPolicy, LookAheadPolicy, SimpleMaxPolicy
    from utils.value_function import CNNValueFun, FFNNValueFun, TabularValueFun
    from algos import FunctionApproximateValueIteration, RNNDemandPredictor, TruePPredictor
    from envs import ASRSEnv, ProbDistEnv, DynamicProbEnv, StaticOrderProcess, SeasonalOrderProcess

    num_products = np.array(eval(args.storage_shape)).prod()
    assert (eval(args.dist_param) is None) or (num_products == len(eval(args.dist_param))), 'storage_shape should be consistent with dist_param length'
    if args.dynamic_order:
        op = SeasonalOrderProcess(num_products = num_products, dist_param = eval(args.dist_param),season_length = 500, beta=1, rho=0.99)
    else:
        op = StaticOrderProcess(num_products = num_products, dist_param = eval(args.dist_param)) 

    base_env = ASRSEnv(eval(args.storage_shape), order_process = op, origin_coord=eval(args.exit_coord))
    if args.true_p:
        true_p = TruePPredictor(base_env, look_back=10, dynamic=args.dynamic_order, init_num_period = args.rnn_init_num_period, num_p_in_states = args.num_p_in_states)
        env = DynamicProbEnv(base_env,demand_predictor = true_p, alpha=1, num_p_in_states = args.num_p_in_states)
    else:
        rnn = RNNDemandPredictor(base_env,look_back=args.rnn_lookback, init_num_period = args.rnn_init_num_period, epochs = args.rnn_epoch)
        env = DynamicProbEnv(base_env,demand_predictor = rnn, alpha=1, num_p_in_states = args.num_p_in_states)

    env_name = env.__name__
    # exp_dir = os.getcwd() + '/data/version4/%s/policy_type%s_envsize_%s_dynamic_%s_p_hat_%s_%s/' % (env_name, args.policy_type,np.array(eval(args.storage_shape)).prod(), args.dynamic_order, not args.true_p, args.exp_name)
    exp_dir = os.getcwd() + '/data/report/%s/policy_type%s_envsize_%s_dynamic_%s_p_hat_%s_%s/' % (env_name, args.policy_type,np.array(eval(args.storage_shape)).prod(), args.dynamic_order, not args.true_p, args.exp_name)
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], level=eval(args.logger_level))
    args_dict = vars(args)
    args_dict['env'] = env_name
    json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)
    if not args.true_p:
        rnn.test_performance_plot(2000, save_to=exp_dir)
        rnn.save(f'{exp_dir}/rnn_model.h5')

    value_fun = FFNNValueFun(env)
    policy = SimpleMaxPolicy(env,
                            value_fun,
                            num_acts = args.num_acts,
                            all_actions= args.all_actions)
    # policy = LookAheadPolicy(env,
    #                         value_fun,
    #                         horizon=args.horizon,
    #                         look_ahead_type=args.policy_type,
    #                         num_acts=args.num_acts)
    if args.dynamic_order:
        last_max_path_length = args.last_max_path_length
    else:
        last_max_path_length = args.max_path_length
        
    algo = FunctionApproximateValueIteration(env,
                            value_fun,
                            policy,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            num_acts=args.num_acts,
                            all_actions= args.all_actions,
                            render=render,
                            num_rollouts = args.num_rollouts,
                            max_itr=args.max_iter,
                            log_itr=5,
                            max_path_length=args.max_path_length,
                            last_max_path_length=last_max_path_length
                            )
    algo.train()
    value_fun.save(f'{exp_dir}/value_fun.h5')

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
    parser.add_argument("--storage_shape", "-s", type=str, default='(2, 5)',
                        help="ASRSEnv storage shape")
    parser.add_argument("--exit_coord", "-e", type=str, default='None',
                        help="Coordinate of the exit, default is the origin")
    parser.add_argument("--dist_param", "-prob", type=str, default='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]',
                        help="ASRSEnv dist_param, the order probability")
    parser.add_argument("--num_rollouts", "-nr", type=int, default=10,
                        help="Number of rollouts used to evaluate policy")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help='Learning rate for training the value function')
    parser.add_argument("--batch_size", "-bs", type=int, default=256,
                        help='batch size for training the value function')
    parser.add_argument("--num_acts", "-a", type=int, default=50,
                        help='Number of actions sampled for maximizing the value function')
    parser.add_argument("--all_actions", "-all_a",  action='store_true',
                        help='Sample all the actions')
    parser.add_argument("--rnn_lookback", type=int, default=1000,
                        help='Number of period lookback for RNN training')
    parser.add_argument("--rnn_init_num_period", type=int, default=10000,
                        help='Number of initial period used to train RNN')
    parser.add_argument("--max_path_length", "-mp", type=int, default=10,                       
                        help='Number of period used for rollout')
    parser.add_argument("--last_max_path_length", "-lmp", type=int, default=10,
                        help='Number of period used for final video')
    parser.add_argument("--rnn_epoch", type=int, default=2,
                        help='Number of epoch used to train RNN')
    parser.add_argument("--num_p_in_states", type=int, default=10,
                        help='Number of period of p used in value function')
    parser.add_argument("--dynamic_order", "-d", action='store_true', help="Use dynamic order process")
    parser.add_argument("--true_p", "-tp", action='store_true', help="Use real p")
    parser.add_argument("--exp_name", type=str,default='', help="Experiment name")

    args = parser.parse_args()
    main(args)

