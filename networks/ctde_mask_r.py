"""Customize the centralized critic model
Reference:
https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py
https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
THis policy is prepared for SumoEnvMulti_md_action_mask.py.
"""

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

import torch as th
from torch import nn
from utils.sig_config import sig_configs

agent_ids = sig_configs['h_corridor']['sig_ids']


class CTDEActionMask(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        n_input_channels = 2
        # print(obs_space, act_space)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward
        # with th.no_grad():
        #     n_flatten = self.cnn(
        #         th.as_tensor(obs_space.sample()['local'][None]).float()
        #     ).shape[1]
        # print(n_flatten)
        n_flatten = 3008
        # print(num_outputs)
        num_outputs = 48

        self.policy_fn = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

        self.value_fn = nn.Sequential(
            nn.Linear((n_flatten+10)*5, 128),
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
        # print(f'-----------input_dict: {input_dict["obs"]}, {input_dict["obs"].keys()}-------------')
        # print()
        # print(f'-----------agent index:, {input_dict["agent_index"]}, {input_dict}---------------')
        # print()
        # Store model-input for possible `value_function()` call.
        # self._model_in = [input_dict['obs_flat'], state, seq_lens]
        # Input all agent's obs and calculate the centralized value
        # central_out_lst = [self.cnn(input_dict['obs']['global'][agent]) for agent in agent_ids]
        central_out_lst = []
        for agent in agent_ids:
            central_out_lst.append(self.cnn(input_dict['obs']['global'][agent]))
            central_out_lst.append(input_dict['obs']['phases'][agent])
        central_out = torch.cat(central_out_lst, 1)
        # print('------central out:', type(central_out), central_out.shape)
        self._value_out = self.value_fn(central_out)
        # Input the agent's own obs
        # self.agent_out = self.cnn(input_dict['obs']['own_obs'].to(th.float))

        self.agent_out = self.cnn(input_dict['obs']['local'])
        # print(self.agent_out.size())

        # Extract the available actions tensor from the observation.
        action_mask = input_dict['obs']['action_mask']
        # print(f'---action mask:\n {action_mask.size()}')

        # Compute the unmasked logits.
        logits = self.policy_fn(self.agent_out)

        # if 0 in action_mask:
        # print(f'----Original logits:\n {logits.size()}')

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # print(inf_mask.size())
        masked_logits = logits + inf_mask
        # print(f'---Masked logits"\n {masked_logits}')
        return masked_logits, state

        # else:
        #     return logits, state

    def value_function(self):
        # value_out = self.value_fn(self.central_out)
        return self._value_out.flatten()  # Flatten to ensure vf_preds.shape == rewards.shape
