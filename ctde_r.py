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
from environments.SumoEnvMulti_ctde_r import SumoEnvMulti
from networks.centralized_critic_policy_input_action import CentralizedCriticModel
from utils.reward_normalization import RewardNormalization
from utils.sig_config import sig_configs

agent_ids = sig_configs['r_corridor']['sig_ids']


def sumo_cmd(gui=False, log=False):
    sumoBinary = '/usr/share/sumo/bin/sumo-gui' if gui else '/usr/share/sumo/bin/sumo'

    # choose scenario
    data_path = '/home/ytj/PycharmProjects/MARL_TSC/scenarios/r_corridor'

    if log:
        cmd = [
            sumoBinary, "-c", f'{data_path}/r.sumocfg', '--time-to-teleport', '-1',
            '--no-warnings', '--random', '--no-step-log',
            "--duration-log.statistics",
        ]
    else:
        cmd = [
            sumoBinary, "-c", f'{data_path}/r.sumocfg', '--time-to-teleport', '-1',
            '--no-warnings', '--random', '--no-step-log',
        ]
    return cmd


def env_creator(args):
    cmd = sumo_cmd()
    env = SumoEnvMulti(cmd)
    return ParallelPettingZooEnv(env)


def train_ctde():
    stop = {'episodes_total': 1000}

    log_dir = '/home/ytj/PycharmProjects/MARL_TSC/logs/r_bus/offpeak/ctde/no_md'

    env = env_creator({})
    register_env('sumo_env', env_creator)
    # Register policy
    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)

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
            # observation_filter='MeanStdFilter',
            # batch_mode="complete_episodes",
            num_rollout_workers=8,
            enable_connectors=False,
        )
        # .callbacks(FillInActions)
        .training(
            model={"custom_model": "CustomModel"},
            # use_kl_loss=False,
            # kl_coeff=0.2,
            kl_coeff=tune.grid_search([0.0, 0.2]),
            vf_share_layers=False,
            gamma=0.8,
            # gamma=tune.grid_search([0.7, 0.75, 0.8, 0.85]),
            lr_schedule=[[0, 0.001], [2e6, 0.0005]],
            # lr=0.001,
            # lr=tune.grid_search([0.005, 0.003, 0.001]),
            use_gae=True,
            lambda_=0.99,
            # lambda_=tune.grid_search([0.99]),
            train_batch_size=2048,
            sgd_minibatch_size=256,
            # sgd_minibatch_size=tune.grid_search([256, 512]),
            num_sgd_iter=3,
            # num_sgd_iter=tune.grid_search([3, 5]),
            vf_loss_coeff=0.5,
            # vf_loss_coeff=tune.grid_search([0.1, 0.01, 0.001, 0]),
            entropy_coeff=0.01,
            # entropy_coeff_schedule=[[0, 0.01], [2e6, 0.001]],
            # entropy_coeff=tune.grid_search([0.1, 0.05]),
            clip_param=0.2,
            # clip_param=tune.grid_search([0.1, 0.2]),
            grad_clip=0.5,
        )
        # Independent learning
        # .multi_agent(
        #     policies=env.get_agent_ids(),
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        # )
        # Partially parameter sharing
        .multi_agent(
            policies={'policy1', 'policy2'},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: 'policy1'
            if agent_id == 'h2'
            else 'policy2',
        )
        # Parameter sharing
        # .multi_agent(
        #     policies={'shared_policy'},
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: 'shared_policy'),
        # )
        .resources(num_gpus=1)
        .reporting(
            # keep_per_episode_custom_metrics=True,
            metrics_num_episodes_for_smoothing=50,
        )
    )

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            num_samples=4,
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


def test_ctde(policy_path, gui=True, log=True):
    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)
    checkpoint_path = policy_path

    sumo_cmd = set_sumo(
        gui=gui,
        sumocfg_path='/home/ytj/PycharmProjects/MARL_TSC/scenarios/r_corridor/r.sumocfg',
        log_path=checkpoint_path,
    )
    # print(sumo_cmd)
    env = SumoEnvMulti(sumo_cmd)

    # Independent policy
    # policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in agent_ids}
    # Partially shared policy
    policies = {a: Policy.from_checkpoint(checkpoint_path)['policy1' if a == 'h2' else 'policy2']
                for a in agent_ids}
    # Shared policy
    # policy = Policy.from_checkpoint(checkpoint_path)['shared_policy']
    # print(policies)

    obs, _ = env.reset()
    terminations = {}
    while True not in terminations.values():
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy = policies[agent_id]  # Independent policies
            actions[agent_id] = policy.compute_single_action(agent_obs, explore=False)[0]
        obs, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


def evaluate_ctde(policy_path):
    start_time = time.time()

    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)
    checkpoint_path = policy_path

    result_path = checkpoint_path
    create_result_folder(result_path)

    # Independent policy
    # policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in agent_ids}
    # Partially shared policy
    policies = {a: Policy.from_checkpoint(checkpoint_path)['policy1' if a == 'h2' else 'policy2']
                for a in agent_ids}
    # Shared policy
    # policy = Policy.from_checkpoint(checkpoint_path)['shared_policy']
    # print(policies)

    tot_trip_res = []
    # tot_queue_res = []
    # tot_stop_res = {}
    episode = 0
    n = 50
    tot_episode_reward = []
    scenario_path = 'scenarios/r_corridor'
    sumoBinary = checkBinary('sumo')

    while episode < n:
        print(f'------ Episode {str(episode + 1)} of {n} ------', end='\r', flush=True)
        sumo_cmd = [
            sumoBinary, "-c", f'{scenario_path}/r.sumocfg', "--seed", "%d" % episode, '--time-to-teleport', '-1',
            '--no-warnings', '--no-step-log',
            "--tripinfo-output", f'{result_path}/tripinfo.xml',
            # '--queue-output', f'{result_path}/queue.xml',
            # '--stop-output', f'{result_path}/stop.xml',
            # "--duration-log.statistics",
            # "--log", "logfile.xml",
        ]

        env = SumoEnvMulti(sumo_cmd)
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
    for i in range(0, len(trip_mean), 4):
        print(f'trip mean: {trip_mean[i]}, {trip_mean[i + 1]}, {trip_mean[i + 2]}, {trip_mean[i + 3]}')
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
        trail_results.append(evaluate_ctde(policy_path))
        n += 1
    np.savetxt(f'{log_path}/trail_result.csv', trail_results, delimiter=',')
    print(f'Total evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == "__main__":
    # train_ctde()
    # policy_path = 'logs/r_bus/peak/ctde/no_md/PPO_2024-03-31_16-22-30/PPO_sumo_env_67a9d_00002_2_kl_coeff=0.0000_2024-03-31_16-22-32/checkpoint_000058'
    # test_ctde(policy_path)
    # evaluate_ctde(policy_path)
    evaluation('logs/r_bus/offpeak/ctde/no_md/PPO_2024-03-31_23-45-04')
