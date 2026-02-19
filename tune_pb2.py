"""Customizing PPO to leverage a centralized critic.
Reference: https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic_2.py
"""

from ray import air, tune
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.models import ModelCatalog

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environments.SumoEnvMulti_ctde_md_r import SumoEnvMulti
from networks.centralized_critic_policy_input_action import CentralizedCriticModel
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


def tune_ctde():
    stop = {'episodes_total': 1000}

    log_dir = '/home/ytj/PycharmProjects/MARL_TSC/logs/r_bus/peak/ctde/md_far'

    env = env_creator({})
    register_env('sumo_env', env_creator)
    # Register policy
    ModelCatalog.register_custom_model('CustomModel', CentralizedCriticModel)

    pb2 = PB2(
        time_attr='timesteps_total',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=2e6,
        quantile_fraction=0.25,  # copy bottom % with top %
        # Specifies the hyperparameters search space
        hyperparam_bounds={
            'lambda': [0.9, 1.0],
            'lr': [0.0001, 0.01],
            'gamma': [0.6, 0.95],
            # 'train_batch_size': [2048, 40000],
            # 'sgd_minibatch_size': [64, 256],
            # 'num_sgd_iter': [2, 10],
            'clip_param': [0.1, 0.5],
        }
    )

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
            # use_kl_loss=False,
            kl_coeff=0.2,
            vf_share_layers=False,
            # gamma=0.6,
            gamma=tune.grid_search([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
            # lr_schedule=[[0, 0.001], [1e6, 0.0003]],
            # lr=0.0001,
            lr=tune.uniform(0.0001, 0.01),
            use_gae=True,
            lambda_=0.9,
            # lambda_=tune.grid_search([1, 0.99, 0.95]),
            train_batch_size=2048,
            # train_batch_size=tune.grid_search([2048, 4096, 8192]),
            sgd_minibatch_size=256,
            # sgd_minibatch_size=tune.grid_search([128, 256, 512]),
            num_sgd_iter=3,
            # num_sgd_iter=tune.grid_search([3, 5, 7, 9, 10]),
            vf_loss_coeff=0.5,
            # vf_loss_coeff=tune.grid_search([0.1, 0.01, 0.001, 0]),
            entropy_coeff=0.01,
            # entropy_coeff_schedule=[[0, 0.01], [1e6, 0.001]],
            # entropy_coeff=tune.grid_search([0.1, 0.05]),
            clip_param=0.1,
            # clip_param=tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
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
            scheduler=pb2,
            num_samples=4,
            # reuse_actors=True,
        ),
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=5,
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


if __name__ == "__main__":
    tune_ctde()