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

from typing import Optional

import acme
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents.tf.nfsp import acting
from acme.agents.tf.nfsp import learning
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class NFSP(acme.Actor):
  """NFSP Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      rl_network: snt.Module,
      sl_network: snt.Module,
      replay_buffer_capacity: int,
      reservoir_buffer_capacity: int,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      batch_size: int = 64,
      rl_learning_rate: float = 1e-3,
      sl_learning_rate: float = 1e-3,
      anticipatory_param: float = 0.1,
  ):
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    # TODO create second replay table
    rl_replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=rl_max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(environment_spec))
    self._rl_server = reverb.Server([rl_replay_table], port=None)

    # The adder is used to insert observations into replay.
    rl_address = f'localhost:{self._rl_server.port}'
    rl_adder = adders.NStepTransitionAdder(
        client=reverb.Client(rl_address),
        n_step=n_step,
        discount=rl_discount)

    # The dataset provides an interface to sample from replay.
    rl_replay_client = reverb.TFClient(rl_address)
    rl_dataset = datasets.make_reverb_dataset(
        server_address=rl_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)


    # TODO check this
    sl_replay_table = reverb.Table(
        name='sl_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Uniform(),
        max_size=sl_max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1))
        #signature=adders.NStepTransitionAdder.signature(environment_spec)) # TODO
    self._sl_server = reverb.Server([sl_replay_table], port=None)

    # The adder is used to insert observations into replay.
    sl_address = f'localhost:{self._sl_server.port}'
    # TODO almost def doesn't make sense to use NStep here
    sl_adder = adders.NStepTransitionAdder(
        client=reverb.Client(sl_address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    sl_replay_client = reverb.TFClient(sl_address)
    sl_dataset = datasets.make_reverb_dataset(
        server_address=sl_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # Use constant 0.05 epsilon greedy policy by default.
    if epsilon is None:
      epsilon = tf.Variable(0.05, trainable=False)
    if policy_network is None:
      policy_network = snt.Sequential([
          network,
          lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
      ])

    tf2_utils.create_variables(rl_network, [environment_spec.observations])
    tf2_utils.create_variables(sl_network, [environment_spec.observations])

    self._actor = acting.NFSPActor(rl_network,
                                   sl_network,
                                   sl_adder,
                                   sl_adder)



    # Create a target network.
    target_network = copy.deepcopy(network)

    # TODO import
    self._rl_learner = learning.DQNLearner(
        network=rl_network,
        target_network=target_network,
        discount=rl_discount,
        importance_sampling_exponent=rl_importance_sampling_exponent,
        learning_rate=rl_learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        logger=logger,
        checkpoint=checkpoint)


    # TODO import BCLearner
    # TODO counter?
    self._sl_learner = learning.BCLearner(
        network=sl_network,
        learning_rate=sl_learning_rate,
        dataset=sl_dataset,
        counter=learner_counter)


  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

  # TODO
  def update(self, wait: bool = False):
    # Run a number of learner steps (usually gradient steps).
    while self._can_sample():
      self._learner.step()

  def select_action(self, observation: np.ndarray) -> int:
    return self._actor.select_action(observation)
