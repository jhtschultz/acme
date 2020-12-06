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

"""NFSP actor implementation."""

import enum

from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

MODE = enum.Enum("mode", "best_response average_policy")


class NFSPActor(core.Actor):
  """An NFSP actor that switches between exploitative and average strategy."""

  def __init__(
      self,
      rl_network: snt.Module,
      sl_network: snt.Module,
      anticipatory_param: float,
      reservoir_buffer_capacity: int,
      rl_adder: adders.Adder = None,
      sl_adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
  ):

    self._anticipatory_param = anticipatory_param

    self._rl_adder = rl_adder
    self._sl_adder = sl_adder
    self._variable_client = variable_client
    # TODO do we need this?
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._rl_network = rl_network
    self._sl_network = sl_network

    # TODO need to use a counter so we can save/restore
    # TODO like a real counter (see DQN learner)
    # We only increment counter when there's an observation
    self._step_counter = 0


  @tf.function
  def _rl_policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._rl_network(batched_observation)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action

  @tf.function
  def _sl_policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._sl_network(batched_observation)
    policy = tfd.Categorical(probs=policy)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action

  @tf.function
  def _get_avg_policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._sl_network(batched_observation)

    return policy


  # TODO should this be a tensorflow op?
  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Pass the observation through the appropriate policy network.
    if self._mode == MODE.best_response:
      action = self._rl_policy(observation)
    elif self._mode == MODE.average_policy:
      action = self._sl_policy(observation)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(action)

  # TODO check this
  # TODO Reservoir sampling for sl_adder
  # TODO handle update epsilon properly
  def observe_first(self, timestep: dm_env.TimeStep):
    self._step_counter += 1
    self._rl_network._layers[-1].update_epsilon(0.001 + (0.06 - 0.001) *
        (1 - self._step_counter / 10000000)**1.0)
    #self._rl_network.update_epsilon(
    #    self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
    #    (1 - decay_steps / self._epsilon_decay_duration)**power)
    self._sample_episode_policy()
    if self._rl_adder:
      self._rl_adder.add_first(timestep)
    if self._mode == MODE.best_response and self._sl_adder:
      self._sl_adder.add_first(timestep)

  # TODO check this
  # TODO !! Reservoir sampling! Only add randint() < buffersize
  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._step_counter += 1
    if self._rl_adder:
      self._rl_adder.add(action, next_timestep)
    if self._mode == MODE.best_response and self._sl_adder:
      self._sl_adder.add(action, next_timestep)

  # TODO only used for distributed??
  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)
