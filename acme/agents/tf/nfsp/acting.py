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
      sl_network: snt.Module,
      rl_network: snt.Module,
      sl_adder: adders.Adder = None,
      rl_adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
  ):

    # Store these for later use.
    self._sl_adder = sl_adder
    self._rl_adder = rl_adder
    self._variable_client = variable_client
    self._sl_network = sl_network
    self._rl_network = rl_network

    # TODO need to apply mask to sl network
    self._sl_policy = tf.function(sl_network)
    self._rl_policy = tf.function(rl_network)

  # TODO should this be a tensorflow op?
  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Pass the observation through the appropriate policy network.
    if self._mode == MODE.best_response:
      action = self._rl_policy(observervation)
    elif self._mode == MODE.average_policy:
      action = self._sl_policy(observervation)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(action)

  # TODO check this
  # TODO Reservoir sampling for sl_adder
  def observe_first(self, timestep: dm_env.TimeStep):
    self._sample_episode_policy()
    if self._rl_adder:
      self._rl_adder.add_first(timestep)
    if self._mode == MODE.best_response and self._sl_adder:
      self._sl_adder.add_first(timestep)

  # TODO check this
  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._rl_adder:
      self._rl_adder.add(action, timestep)
    if self._mode == MODE.best_response and self._sl_adder:
      self._sl_adder.add(action, timestep)

  # TODO only used for distributed??
  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)
