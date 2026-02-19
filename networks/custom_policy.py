from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch as th
from torch import nn


class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        n_input_channels = obs_space['local'].shape[0]
        # print(obs_space)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # print(self.cnn)
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(obs_space.sample()['local'][None]).float()
            ).shape[1]

        self.policy_fn = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

        self.value_fn = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._value_out = None
        # print(self.policy_fn, self.value_fn)

    def forward(self, input_dict, state, seq_lens):
        # print("Input dict:", input_dict, type(input_dict))
        # print()
        # print("The unpacked input tensors:", input_dict["rewards"], input_dict['rewards'].size())
        # print()
        model_out = self.cnn(input_dict['obs']['local'])  # Convert the input tensor type to float
        # print(model_out, model_out.size())
        # print()
        self._value_out = self.value_fn(model_out)
        # print('value out:', self._value_out, self._value_out.size())
        return self.policy_fn(model_out), []

    def value_function(self):
        return self._value_out.flatten()  # Flatten to ensure vf_preds.shape == rewards.shape
