# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Fictitious Self-Play (NFSP) agent implementation."""

import copy
from typing import Optional

import acme
from acme import datasets
from acme import specs
from acme import types
from acme import wrappers
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf.nfsp_bc import learning as bc_learning
from acme.agents.tf.nfsp_dqn import learning as dqn_learning
from acme.agents.tf.nfsp import acting
from acme.agents.tf.nfsp import learning as nfsp_learning
from acme.tf import utils as tf2_utils
from acme.tf.networks import legal_actions
from acme.utils import counting
from acme.utils import loggers
import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class NFSP(agent.Agent):
  """NFSP Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      rl_network: snt.Module,
      bc_network: snt.Module,
      replay_buffer_capacity: int,
      reservoir_buffer_capacity: int,
      discount: float = 1.0,
      batch_size: int = 128,
      rl_learning_rate: float = 0.01,
      bc_learning_rate: float = 0.01,
      anticipatory_param: float = 0.1,
      prefetch_size: int = 4,
      target_update_period: int = 300,
      samples_per_insert: float = 32.0,  # TODO unused
      min_replay_size: int = 1000,
      observations_per_step: int = 64,
      importance_sampling_exponent: float = 0.2,  # TODO currently have this commented out in DQN learner
      priority_exponent: float = 0.0,
      n_step: int = 1,
      epsilon: Optional[tf.Tensor] = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      checkpoint_subpath: str = '~/acme/',
  ):
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    # TODO create second replay table
    rl_replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        #sampler=reverb.selectors.Prioritized(priority_exponent),  # TODO
        remover=reverb.selectors.Fifo(),
        max_size=replay_buffer_capacity,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(environment_spec))
    self._rl_server = reverb.Server([rl_replay_table], port=None)
    print("RL_PORT: ", self._rl_server.port)

    # The adder is used to insert observations into replay.
    rl_address = f'localhost:{self._rl_server.port}'
    rl_adder = adders.NStepTransitionAdder(
        client=reverb.Client(rl_address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    rl_replay_client = reverb.TFClient(rl_address)

    # TODO we're hacking this for now
    self._rl_buffer_client = reverb.Client(rl_address)

    rl_dataset = datasets.make_reverb_dataset(
        server_address=rl_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)


    # TODO check this
    bc_replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Uniform(),
        max_size=reservoir_buffer_capacity,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(environment_spec)) # TODO
        #signature=adders.ReverbAdder.signature(environment_spec))  # TODO
    self._bc_server = reverb.Server([bc_replay_table], port=None)
    print("BC_PORT: ", self._bc_server.port)

    # The adder is used to insert observations into replay.
    bc_address = f'localhost:{self._bc_server.port}'
    # TODO should we be using NStep here?
    bc_adder = adders.NStepTransitionAdder(
        client=reverb.Client(bc_address),
        n_step=1,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    # TODO remove bc_replay_client - only needed for DQN to update priorities
    bc_replay_client = reverb.TFClient(bc_address)
    # TODO we're hacking this for now
    self._bc_buffer_client = reverb.Client(bc_address)
    bc_dataset = datasets.make_reverb_dataset(
        server_address=bc_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # TODO
    # Use constant 0.05 epsilon greedy policy by default.
    #if epsilon is None:
    #  epsilon = tf.Variable(0.05, trainable=False)
    rl_policy_network = snt.Sequential(
        [rl_network, legal_actions.EpsilonGreedy(epsilon=0.1, threshold=-1e8)])
    # TODO
    bc_policy_network = bc_network

    tf2_utils.create_variables(rl_network, [environment_spec.observations])
    tf2_utils.create_variables(bc_network, [environment_spec.observations])

    actor = acting.NFSPActor(rl_policy_network,
                             bc_policy_network,
                             anticipatory_param,
                             reservoir_buffer_capacity,
                             rl_adder,
                             bc_adder)



    # Create a target network.
    target_network = copy.deepcopy(rl_network)
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    # TODO import
    rl_learner = dqn_learning.DQNLearner(
        network=rl_network,
        target_network=target_network,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=rl_learning_rate,
        target_update_period=target_update_period,
        dataset=rl_dataset,
        replay_client=rl_replay_client,
        logger=logger,
        checkpoint=checkpoint)


    # TODO counter?
    bc_learner = bc_learning.BCLearner(
        network=bc_network,
        learning_rate=bc_learning_rate,
        dataset=bc_dataset)
        #counter=learner_counter) TODO counter created by default?

    learner = nfsp_learning.NFSPLearner(rl_learner, bc_learner, self._rl_buffer_client, self._bc_buffer_client)

    # TODO add back checkpoint
    #if checkpoint:
    #  self._checkpointer = tf2_savers.Checkpointer(
    #      directory=checkpoint_subpath,
    #      objects_to_save=learner.state,
    #      subdirectory='dqn_learner',
    #      time_delta_minutes=60.)
    #else:
    #  self._checkpointer = None

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=observations_per_step)
        # TODO
        #observations_per_step=float(batch_size) / samples_per_insert)



  # TODO checkpoint
  # TODO might want to override this
  def update(self):
    super().update()
    #if self._checkpointer is not None:
    #  self._checkpointer.save()

