from environments.SumoEnvMulti2 import SumoEnvMulti

import argparse
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from utils.utils import *
from networks.custom_policy import CustomCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('-s', '--scenario', type=str, default='corridor')
    ap.add_argument('--log', action='store_true')
    args = ap.parse_args()

    if args.test:
        test_rllib(args)
    else:
        train_rllib()


def sumo_cmd(gui=False, scenario='corridor', log=False):
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # choose scenario
    if scenario == 'intersection':
        data_path = 'scenarios/h_intersection'
    elif scenario == 'corridor':
        data_path = 'scenarios/h_corridor'
    else:
        data_path = 'scenarios/r_intersection'

    if log:
        cmd = [
            sumoBinary, "-c", f'{data_path}/h.sumocfg',
            '--no-warnings', '--random', '--no-step-log',
            "--duration-log.statistics",
        ]
    else:
        cmd = [
            sumoBinary, "-c", f'{data_path}/h.sumocfg',
            '--no-warnings', '--random', '--no-step-log',
        ]
    return cmd


def env_creator(args):
    cmd = sumo_cmd()
    env = SumoEnvMulti(cmd)
    return ParallelPettingZooEnv(env)


def train_rllib():
    # episodes_total, timesteps_total,training_iteration
    stop = {'episodes_total': 1000}

    log_dir = '/home/ytj/PycharmProjects/MARL_TSC/logs/PPO_PS/850'

    env = env_creator({})
    register_env('sumo_env', env_creator)
    # Register policy
    ModelCatalog.register_custom_model('CustomCNN', CustomCNN)

    config = (
        PPOConfig()
        .environment('sumo_env')
        .resources(num_gpus=1)
        .rollouts(
            num_rollout_workers=8,
        )
        # # Independent learning
        # .multi_agent(
        #     policies=env.get_agent_ids(),
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        #     count_steps_by='agent_steps',
        # )
        # Parameter sharing
        .multi_agent(
            policies={'shared_policy'},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: 'shared_policy'),
            # count_steps_by='agent_steps',
        )
        .training(
            gamma=0.65,
            # gamma=tune.grid_search([0.6, 0.7]),
            model={'custom_model': 'CustomCNN'},
            lr_schedule=[[0, 0.001], [1e6, 0.0001]],
            # lr=0.0001,
            # lr=tune.grid_search([0.0001, 0.0003, 0.0005]),
            use_gae=True,
            lambda_=0.95,
            sgd_minibatch_size=256,
            # sgd_minibatch_size=tune.grid_search([256, 512]),
            num_sgd_iter=5,
            # num_sgd_iter=tune.grid_search([5, 10, 20]),
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            clip_param=0.2,
            # clip_param=tune.grid_search([0.1, 0.2, 0.3]),
            grad_clip=0.5,
        )
        .framework(framework='torch')
    )

    tuner = tune.Tuner(
        'PPO',
        tune_config=tune.TuneConfig(
            metric='episode_reward_mean',
            mode='max',
        ),
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=500,
                checkpoint_at_end=True
            ),
            local_dir=log_dir
        ),
        param_space=config.to_dict(),
    )
    result = tuner.fit()
    best_trial = result.get_best_result()
    # print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial episode_reward_mean: {best_trial.last_result['episode_reward_mean']}")
    print(f"Best trial path: {best_trial.path}")


def test_rllib(args):
    gui = True if args.gui else False
    log = True if args.log else False
    cmd = sumo_cmd(gui=gui, log=log)
    env = SumoEnvMulti(cmd)
    ModelCatalog.register_custom_model('CustomCNN', CustomCNN)
    checkpoint_path = ('logs/PPO_PS/850/PPO_2024-01-03_14-43-45/PPO_sumo_env_69149_00000_0_2024-01-03_14-43-47'
                       '/checkpoint_000000')

    # Independent policies
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


if __name__ == '__main__':
    main()




