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

"""Example running DQN on OpenSpiel game in a single process."""

from absl import app
from absl import flags

import acme
from acme import core
from acme import wrappers
from acme.agents.tf import nfsp
from acme.environment_loops import nfsp_environment_loop
from acme.tf.networks import legal_actions
from acme.wrappers import open_spiel_wrapper
from open_spiel.python import rl_environment
import numpy as np  # TODO where to place this?
import sonnet as snt
import tensorflow as tf

#flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
#flags.DEFINE_integer("num_players", 2, "Number of players")

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "kuhn_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
#flags.DEFINE_list("hidden_layers_sizes", [
#    128,
#], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
#flags.DEFINE_string("optimizer_str", "sgd",
#                    "Optimizer, choose from 'adam', 'sgd'.")
#flags.DEFINE_string("loss_str", "mse",
#                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 300,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
#flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
#                     "Number of game steps over which epsilon is decayed.")
#flags.DEFINE_float("epsilon_start", 0.06,
#                   "Starting exploration parameter.")
#flags.DEFINE_float("epsilon_end", 0.001,
#                   "Final exploration parameter.")
flags.DEFINE_string("evaluation_metric", "nash_conv",
                    "Choose from 'exploitability', 'nash_conv'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/tmp/nfsp_test",
                    "Directory to save/load the agent.")


class RandomActor(core.Actor):
  def select_action(self, observation):
    legals = np.squeeze(np.nonzero(observation.legal_actions))
    return np.random.choice(legals)

  def observe_first(self, timestep):
    pass

  def observe(self, action, next_timestep):
    pass

  def update(self, wait = False):
    pass
    

def _make_rl_network(environment_spec):
  return legal_actions.MaskedSequential([
      snt.Flatten(),
      snt.nets.MLP([128, environment_spec.actions.num_values])
  ])

# TODO this will eventually crash
def _make_sl_network(environment_spec):
  return snt.Sequential([legal_actions.MaskedSequential([
      snt.Flatten(),
      snt.nets.MLP([128, environment_spec.actions.num_values])]),
      ])
      #lambda logits: tf.nn.softmax(logits)])

def main(_):
  # Create an environment and grab the spec.
  env_configs = {"players": FLAGS.num_players} if FLAGS.num_players else {}
  raw_environment = rl_environment.Environment(FLAGS.game_name, **env_configs)

  environment = open_spiel_wrapper.OpenSpielWrapper(raw_environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  environment_spec = acme.make_environment_spec(environment)



  # Construct the agents.
  agents = []

  for i in range(environment.num_players):
    agents.append(
        nfsp.NFSP(
            environment_spec=environment_spec,
            rl_network=_make_rl_network(environment_spec),
            sl_network=_make_sl_network(environment_spec),
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            reservoir_buffer_capacity=FLAGS.reservoir_buffer_capacity,
            discount=FLAGS.discount_factor,
            batch_size=FLAGS.batch_size,
            anticipatory_param=FLAGS.anticipatory_param,
            prefetch_size=4,
            target_update_period=FLAGS.update_target_network_every,
            samples_per_insert=32.0,
            min_replay_size=FLAGS.min_buffer_size_to_learn,
            observations_per_step=FLAGS.learn_every,
            importance_sampling_exponent=0.2,  # TODO currently have this commented out in DQN learner
            priority_exponent=0.0,
            n_step=1,
            epsilon=None,  # TODO need to add epsilon start end
            logger=None,
            checkpoint=FLAGS.use_checkpoints,
            checkpoint_subpath=FLAGS.checkpoint_dir,
            ))
  # TODO
  #agents[1] = RandomActor()

  # Run the environment loop.
  loop = nfsp_environment_loop.NFSPEnvironmentLoop(
      environment,
      agents,
      eval_every=FLAGS.eval_every,
      evaluation_metric=FLAGS.evaluation_metric,
      )
  loop.run(num_episodes=FLAGS.num_train_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
