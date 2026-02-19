"""Customize the centralized critic model
Reference: https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py
THis policy is prepared for SumoEnvMulti_ctde_input_action.py.
"""

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch as th
from torch import nn
from utils.sig_config import sig_configs

agent_ids = sig_configs['h_corridor']['sig_ids']


class CentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        n_input_channels = obs_space['local'].shape[0]
        # print(n_input_channels)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # This is for the first intersection with obs shape of 2*50*5
        self.cnn2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 1), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward
        # with th.no_grad():
        #     n_flatten = self.cnn1(
        #         th.as_tensor(obs_space.sample()['local'][None]).float()
        #     ).shape[1]
        # print(n_flatten)

        self.policy_fn1 = nn.Sequential(
            nn.Linear(3008, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

        self.policy_fn2 = nn.Sequential(
            nn.Linear(3008, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

        self.value_fn = nn.Sequential(
            nn.Linear((3008+10)*5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # self._model_in = None
        # self.central_out = None
        self.agent_out = None
        self._value_out = None
        # print(self.policy_fn, self.value_fn)

    def forward(self, input_dict, state, seq_lens):
        # original_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, 'torch')
        # print(f'-----------input_dict:, {input_dict}, {input_dict["obs"]["local"].size()[-1]}---------------')
        # print()
        # print(f'-----------state: {state}, {type(state)}-------------')
        # print()
        # print(f'-----------seq_lens:, {seq_lens}, {type(seq_lens)}---------------')
        # print()
        # Store model-input for possible `value_function()` call.
        # self._model_in = [input_dict['obs_flat'], state, seq_lens]
        # Input all agent's obs and calculate the centralized value
        # central_out_lst = [self.cnn(input_dict['obs']['global'][agent]) for agent in agent_ids]
        central_out_lst = []
        for agent in agent_ids:
            if agent == 'h2':
                central_out_lst.append(self.cnn2(input_dict['obs']['global'][agent]))
            else:
                central_out_lst.append(self.cnn1(input_dict['obs']['global'][agent]))
            central_out_lst.append(input_dict['obs']['phases'][agent])
        central_out = torch.cat(central_out_lst, 1)
        # print('------central out:', type(central_out), central_out.shape)
        self._value_out = self.value_fn(central_out)
        # Input the agent's own obs
        # self.agent_out = self.cnn(input_dict['obs']['own_obs'].to(th.float))
        # self.agent_out = self.cnn(input_dict['obs']['local'])
        # if input_dict["obs"]["local"].size()[-1] == 5:
        #     self.agent_out = self.cnn2(input_dict['obs']['local'])
        # else:
        #     self.agent_out = self.cnn1(input_dict['obs']['local'])
        # return self.policy_fn(self.agent_out), []

        if input_dict["obs"]["local"].size()[-1] == 5:
            self.agent_out = self.cnn2(input_dict['obs']['local'])
            return self.policy_fn2(self.agent_out), []
        else:
            self.agent_out = self.cnn1(input_dict['obs']['local'])
            return self.policy_fn1(self.agent_out), []

    def value_function(self):
        # value_out = self.value_fn(self.central_out)
        return self._value_out.flatten()  # Flatten to ensure vf_preds.shape == rewards.shape
