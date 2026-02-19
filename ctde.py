"""Customizing PPO to leverage a centralized critic.
Reference: https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic_2.py
"""

import time
import datetime
import numpy as np
from utils.utils import *
from utils.analysis import analysis_tripinfo, analysis_queue

from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environments.SumoEnvMulti_ctde import SumoEnvMulti
from networks.centralized_critic_policy import CentralizedCriticModel
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


def train_ctde():
    stop = {'episodes_total': 1000}

    log_dir = '/home/ytj/PycharmProjects/MARL_TSC/logs/ctde/850'

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
        .rollouts(
            # batch_mode="complete_episodes",
            num_rollout_workers=8,
            enable_connectors=False,
        )
        # .callbacks(FillInActions)
        .training(
            model={"custom_model": "CustomModel"},
            use_kl_loss=False,
            # kl_coeff=0.0,
            vf_share_layers=False,
            gamma=0.75,
            # gamma=tune.grid_search([0.65, 0.7, 0.75]),
            lr_schedule=[[0, 0.0005], [3e6, 0.00001]],
            # lr=0.0001,
            # lr=tune.grid_search([0.0001, 0.0003, 0.00005]),
            use_gae=True,
            lambda_=1,
            # lambda_=tune.grid_search([1, 0.99]),
            # train_batch_size=2048,
            sgd_minibatch_size=256,
            # sgd_minibatch_size=tune.grid_search([256, 512]),
            num_sgd_iter=3,
            # num_sgd_iter=tune.grid_search([3, 5, 10]),
            vf_loss_coeff=0.5,
            # vf_loss_coeff=tune.grid_search([0.01, 0.001, 0]),
            entropy_coeff=0.05,
            # entropy_coeff=tune.grid_search([0.5, 0.1, 0.05]),
            clip_param=0.2,
            # clip_param=tune.grid_search([0.2, 0.3]),
            # grad_clip=0.5,
        )
        # # Independent learning
        # .multi_agent(
        #     policies=env.get_agent_ids(),
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        #     # count_steps_by='agent_steps',
        # )
        # Parameter sharing
        .multi_agent(
            policies={'shared_policy'},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: 'shared_policy'),
            # observation_fn=central_critic_observer,
        )
        .resources(num_gpus=1)
        # .reporting(
        #     keep_per_episode_custom_metrics=True,
        #     metrics_num_episodes_for_smoothing=1,
        # )
    )

    tuner = tune.Tuner(
        "PPO",
        # tune_config=tune.TuneConfig(
        #     metric='episode_reward_mean',
        #     mode='max',
        # ),
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=500,
                checkpoint_at_end=True
            ),
            stop=stop,
            verbose=1,
            local_dir=log_dir
        ),
    )
    results = tuner.fit()


def test_ctde(policy_path, gui=True, log=True):
    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)
    checkpoint_path = policy_path

    sumo_cmd = set_sumo(
        gui=gui,
        sumocfg_path='/home/ytj/PycharmProjects/MARL_TSC/scenarios/h_corridor/h.sumocfg',
        log_path=checkpoint_path,
    )
    # print(sumo_cmd)
    env = SumoEnvMulti(sumo_cmd)

    # Independent policy
    # policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in env.possible_agents}
    # Shared policy
    policy = Policy.from_checkpoint(checkpoint_path)['shared_policy']
    # print(policies)

    obs, _ = env.reset()
    terminations = {}
    while True not in terminations.values():
        actions = {}
        for agent_id, agent_obs in obs.items():
            # policy = policies[agent_id]  # Independent policies
            actions[agent_id] = policy.compute_single_action(agent_obs)[0]
        obs, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


def evaluate_ctde(policy_path):
    start_time = time.time()

    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)
    checkpoint_path = policy_path

    result_path = checkpoint_path
    create_result_folder(result_path)

    # Independent policy
    # policies = {a: Policy.from_checkpoint(checkpoint_path)[a] for a in env.possible_agents}
    # Shared policy
    policy = Policy.from_checkpoint(checkpoint_path)['shared_policy']
    # print(policies)

    tot_trip_res = []
    tot_queue_res = []
    episode = 0
    n = 50
    scenario_path = 'scenarios/h_corridor'
    sumoBinary = checkBinary('sumo')

    while episode < n:
        print(f'Testing agent\n------ Episode {str(episode + 1)} of {n} ------')
        sumo_cmd = [
            sumoBinary, "-c", f'{scenario_path}/h.sumocfg', "--seed", "%d" % episode, '--time-to-teleport', '-1',
            '--no-warnings', '--no-step-log',
            "--tripinfo-output", f'{result_path}/tripinfo.xml',
            '--queue-output', f'{result_path}/queue.xml',
            # "--duration-log.statistics",
            # "--log", "logfile.xml",
        ]

        env = SumoEnvMulti(sumo_cmd)

        obs, _ = env.reset()
        terminations = {}
        while True not in terminations.values():
            actions = {}
            for agent_id, agent_obs in obs.items():
                # policy = policies[agent_id]  # Independent policies
                actions[agent_id] = policy.compute_single_action(agent_obs)[0]
            obs, rewards, terminations, truncations, infos = env.step(actions)
        env.close()
        episode += 1

        trip_res = analysis_tripinfo(f'{result_path}/tripinfo.xml')
        queue_res = analysis_queue(f'{result_path}/queue.xml')
        tot_trip_res.append(trip_res)
        tot_queue_res.append(queue_res)

    # Change list to array and transpose
    trip_ares = np.array(tot_trip_res)
    queue_ares = np.array(tot_queue_res).T
    # Save to csv files
    np.savetxt(f'{result_path}/trip_result.csv', trip_ares, delimiter=',')
    np.savetxt(f'{result_path}/queue_result.csv', queue_ares, delimiter=',')
    # print(ares)
    print(f'Evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == "__main__":
    # train_ctde()
    policy_path = ('logs/ctde/850/PPO_2024-01-24_23-10-42/PPO_sumo_env_b5dc1_00005_5_clip_param=0.3000,'
                   'entropy_coeff=0.0500_2024-01-24_23-10-44/checkpoint_000001')
    test_ctde(policy_path)
    # evaluate_ctde(policy_path)
