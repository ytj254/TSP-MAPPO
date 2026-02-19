"""Customizing PPO to leverage a centralized critic.
Reference: https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic_2.py
"""

import time
import datetime
import numpy as np
from utils.utils import *
from utils.analysis import analysis_tripinfo

from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environments.SumoEnvMulti_ctde_input_action import SumoEnvMulti
from networks.custom_policy import CustomCNN
from utils.reward_normalization import RewardNormalization
from utils.sig_config import sig_configs

agent_ids = sig_configs['h_corridor']['sig_ids']


def sumo_cmd(gui=False, log=False):
    sumoBinary = '/usr/share/sumo/bin/sumo-gui' if gui else '/usr/share/sumo/bin/sumo'

    # choose scenario
    data_path = '/home/ytj/PycharmProjects/MARL_TSC/scenarios/h_corridor'

    if log:
        cmd = [
            sumoBinary, "-c", f'{data_path}/h.sumocfg', '--time-to-teleport', '-1',
            '--no-warnings', '--random', '--no-step-log',
            "--duration-log.statistics",
        ]
    else:
        cmd = [
            sumoBinary, "-c", f'{data_path}/h.sumocfg', '--time-to-teleport', '-1',
            '--no-warnings', '--random', '--no-step-log',
        ]
    return cmd


def env_creator(args):
    cmd = sumo_cmd()
    env = SumoEnvMulti(cmd)
    return ParallelPettingZooEnv(env)


def train():
    stop = {'episodes_total': 1000}

    log_dir = '/home/ytj/PycharmProjects/MARL_TSC/logs/h_bus/400/ippo'

    env = env_creator({})
    register_env('sumo_env', env_creator)
    # Register policy
    ModelCatalog.register_custom_model('CustomModel', CustomCNN)

    config = (
        PPOConfig()
        .experimental(
            _enable_new_api_stack=False,
            _disable_preprocessor_api=True  # If True, no obs will be preprocessed.
        )
        .environment('sumo_env',
                     # disable_env_checking=True,
                     )
        .framework(framework='torch')
        .callbacks(RewardNormalization)
        .rollouts(
            # batch_mode="complete_episodes",
            num_rollout_workers=8,
            enable_connectors=False,
        )
        # .callbacks(FillInActions)
        .training(
            model={"custom_model": "CustomModel"},
            # use_kl_loss=False,
            kl_coeff=0.0,
            vf_share_layers=False,
            gamma=0.9,
            # gamma=tune.grid_search([0.6, 0.65, 0.7]),
            lr_schedule=[[0, 0.001], [1e6, 0.0003]],
            # lr=0.001,
            # lr=tune.grid_search([0.005, 0.003, 0.001]),
            use_gae=True,
            lambda_=0.95,
            # lambda_=tune.grid_search([1, 0.99, 0.95]),
            train_batch_size=2048,
            # train_batch_size=tune.grid_search([1024, 2048]),
            sgd_minibatch_size=256,
            # sgd_minibatch_size=tune.grid_search([64, 128, 256]),
            num_sgd_iter=3,
            # num_sgd_iter=tune.grid_search([3, 5]),
            vf_loss_coeff=0.5,
            # vf_loss_coeff=tune.grid_search([0.1, 0.01, 0.001, 0]),
            # entropy_coeff=0.01,
            entropy_coeff_schedule=[[0, 0.01], [1e6, 0.001]],
            # entropy_coeff=tune.grid_search([0.5, 0.1, 0.05]),
            clip_param=0.2,
            # clip_param=tune.grid_search([0.1, 0.2]),
            grad_clip=0.5,
        )
        # Independent learning
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            # count_steps_by='agent_steps',
        )
        .resources(num_gpus=1)
        .reporting(
            # keep_per_episode_custom_metrics=True,
            metrics_num_episodes_for_smoothing=50,
        )
    )

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            num_samples=3,
            metric='episode_reward_mean',
            mode='max',
        ),
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=20,
                checkpoint_at_end=True
            ),
            stop=stop,
            verbose=2,
            local_dir=log_dir
        ),
    )
    results = tuner.fit()

    # Get the best checkpoint path
    best_result = results.get_best_result(metric='episode_reward_mean', mode='max', scope='all')
    best_metric = best_result.best_checkpoints[0][1]['episode_reward_mean']
    checkpoint_path = best_result.best_checkpoints[0][0]
    print('The best episode_reward_mean:', best_metric)
    print('The checkpoint for the best result:', checkpoint_path)


def test(policy_path, gui=True, log=True):
    ModelCatalog.register_custom_model('CustomModel', CustomCNN)
    checkpoint_path = policy_path

    sumo_cmd = set_sumo(
        gui=gui,
        sumocfg_path='scenarios/h_corridor/h.sumocfg',
        log_path=checkpoint_path,
    )
    # print(sumo_cmd)
    env = SumoEnvMulti(sumo_cmd)

    # Independent policy
    policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in env.possible_agents}

    obs, _ = env.reset()
    terminations = {}
    while True not in terminations.values():
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy = policies[agent_id]  # Independent policies
            actions[agent_id] = policy.compute_single_action(agent_obs, explore=False)[0]
        obs, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


def evaluate(policy_path):
    start_time = time.time()

    ModelCatalog.register_custom_model('CustomModel', CustomCNN)
    checkpoint_path = policy_path

    result_path = checkpoint_path
    create_result_folder(result_path)

    tot_trip_res = []
    # tot_queue_res = []
    # tot_stop_res = {}
    episode = 0
    n = 50
    tot_episode_reward = []
    scenario_path = 'scenarios/h_corridor'
    sumoBinary = checkBinary('sumo')

    while episode < n:
        # episode_start_time = time.time()
        print(f'------ Episode {str(episode + 1)} of {n} ------', end='\r', flush=True)
        sumo_cmd = [
            sumoBinary, "-c", f'{scenario_path}/h.sumocfg', "--seed", "%d" % episode, '--time-to-teleport', '-1',
            '--no-warnings', '--no-step-log',
            "--tripinfo-output", f'{result_path}/tripinfo.xml',
            # '--queue-output', f'{result_path}/queue.xml',
            # '--stop-output', f'{result_path}/stop.xml',
            # "--duration-log.statistics",
            # "--log", "logfile.xml",
        ]

        env = SumoEnvMulti(sumo_cmd)
        # Independent policy
        policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in env.possible_agents}
        episode_reward = 0
        obs, _ = env.reset()
        terminations = {}
        while True not in terminations.values():
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy = policies[agent_id]  # Independent policies
                actions[agent_id] = policy.compute_single_action(agent_obs, explore=False)[0]
            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += sum(rewards.values())
        env.close()
        # print(f"The {episode}'s reward is: {episode_reward}")
        tot_episode_reward.append(episode_reward)
        trip_res = analysis_tripinfo(f'{result_path}/tripinfo.xml')
        # queue_res = analysis_queue(f'{result_path}/queue.xml')
        # stop_res = analysis_stopinfo(f'{result_path}/stop.xml')

        tot_trip_res.append(trip_res)
        # tot_queue_res.append(queue_res)
        # for stop, headways in stop_res.items():
        #     if stop not in tot_stop_res:
        #         tot_stop_res[stop] = headways
        #     else:
        #         tot_stop_res[stop] += headways
        episode += 1
        # print(f'Episode evaluation time: {datetime.timedelta(seconds=int(time.time() - episode_start_time))}')

    # Change list to array and transpose
    trip_ares = np.array(tot_trip_res)
    # queue_ares = np.array(tot_queue_res).T
    # Save to csv files
    np.savetxt(f'{result_path}/trip_result.csv', trip_ares, delimiter=',')
    # np.savetxt(f'{result_path}/queue_result.csv', queue_ares, delimiter=',')
    # max_len_headways = max(len(v) for v in tot_stop_res.values())
    # np.savetxt(f'{result_path}/stop_result.csv',
    #            np.column_stack([v + [np.nan] * (max_len_headways - len(v)) for v in tot_stop_res.values()]),
    #            delimiter=',', header=','.join(tot_stop_res.keys()), comments='')
    # print(ares)
    trip_mean = np.mean(trip_ares, axis=0)
    [print(f'trip mean: {trip_mean[i: i + 4]}') for i in range(0, len(trip_mean), 4)]
    print(f'episode_reward_mean: {np.mean(tot_episode_reward)}')
    print(f'episode_reward_max: {max(tot_episode_reward)}')
    print(f'episode_reward_min: {min(tot_episode_reward)}')
    print(f'Evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')
    return trip_mean


def evaluation(log_path):
    start_time = time.time()
    trail_results = []
    folder_list = list_folders(log_path)
    n = 1
    for folder in folder_list:
        print()
        print(f'------ Folder name: {folder} of {len(folder_list)} ------')
        policy_folders = list_folders(os.path.join(log_path, folder))
        # Get the policy folder with the max number
        policy_folder = max(policy_folders, key=lambda x: int(x.split('_')[1]))
        policy_path = os.path.join(log_path, folder, policy_folder)
        # trail_results.append(folder + evaluate_ctde(policy_path))
        trail_results.append(evaluate(policy_path))
        n += 1
    np.savetxt(f'{log_path}/trail_result.csv', trail_results, delimiter=',')
    print(f'Total evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == "__main__":
    # train()
    # policy_path = 'logs/h_bus/850/ippo/PPO_2024-04-24_09-57-48/PPO_sumo_env_a361a_00000_0_2024-04-24_09-57-50/checkpoint_000092'
    # test(policy_path)
    # evaluate(policy_path)
    evaluation('logs/h_bus/400/ippo/PPO_2024-05-21_10-48-06')
