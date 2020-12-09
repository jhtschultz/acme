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

"""Learner for the NFSP agent."""

import time
from typing import Dict, List, Mapping, Optional

import acme
from acme import specs
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions


class NFSPLearner(acme.Learner, tf2_savers.TFSaveable):
  """Learner for NFSP."""

  def __init__(
      self,
      rl_learner: acme.Learner,
      sl_learner: acme.Learner,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
  ):

    self._rl_learner = rl_learner
    self._sl_learner = sl_learner

    # Internalise, optimizer, and dataset.
    # TODO combine here?
    #self._variables = network.variables

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Create a snapshotter object.
    if checkpoint:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'rl_network': rl_learner._network,
                           'sl_network': sl_learner._network},
                                                 time_delta_minutes=60.)
    else:
      self._snapshotter = None

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Do a batch of SGD.
    result = {'rl_loss': self._rl_learner._step()['loss'],
              'sl_loss': self._sl_learner._step()['loss']}

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    result.update(counts)

    # Snapshot and attempt to write logs.
    if self._snapshotter is not None:
      self._snapshotter.save()
    #self._logger.write(result)

  # TODO get from both learners - DONE?
  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy([self._rl_learner._variables, self._sl_learner._variables])

  # TODO get from both learners - DONE
  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'rl_network': self._rl_learner._network,
        'rl_target_network': self._rl_learner._target_network,
        'rl_optimizer': self._rl_learner._optimizer,
        'sl_network': self._sl_learner._network,
        'sl_optimizer': self._sl_learner._optimizer,
        'num_steps': self._num_steps
    }

