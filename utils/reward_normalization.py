import numpy as np
from collections import deque
from typing import Dict, Tuple

from ray.rllib import SampleBatch, RolloutWorker
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.policy import Policy


class FIFOContainer:
    shared_queue = deque(maxlen=5000)

    @classmethod
    def add_array(cls, array):
        for reward in array:
            cls.shared_queue.append(reward)

    @classmethod
    def mean(cls):
        if not cls.shared_queue:
            return None
        return np.mean(cls.shared_queue)

    @classmethod
    def variance(cls):
        if not cls.shared_queue:
            return None
        return np.var(cls.shared_queue)

    @classmethod
    def get_num_elements(cls):
        return len(cls.shared_queue)


class RewardNormalization(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        # postprocessed_batch = standardize_fields(postprocessed_batch, ['rewards'])
        running_stats = FIFOContainer()
        running_stats.add_array(postprocessed_batch['rewards'])
        # print(running_stats.get_num_elements())
        # print(running_stats.mean())
        # print(running_stats.variance())
        # Normalize rewards
        postprocessed_batch['rewards'] = (postprocessed_batch['rewards'] - running_stats.mean()) / (running_stats.variance() + 1e-10)
        if postprocessed_batch[SampleBatch.TERMINATEDS][-1]:
            last_r = 0.0
        else:
            last_r = postprocessed_batch[SampleBatch.VF_PREDS][-1]
        postprocessed_batch = compute_advantages(
            postprocessed_batch,
            last_r,
            policies[policy_id].config['gamma'],
            policies[policy_id].config['lambda'],
            use_gae=policies[policy_id].config['use_gae'],
        )
        # print(postprocessed_batch)
        # print('Rewards:', postprocessed_batch['rewards'], type(postprocessed_batch['rewards']))
        # print('Advantages:', postprocessed_batch['advantages'], type(postprocessed_batch['advantages']))
        # print('Original_batches:', original_batches)