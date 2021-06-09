# from causal_graph import CausalGraph
from knowledge import Knowledge
import util
import gutil
import random
from environment import Environment
from assignment_models import AssignmentModel, discrete_model
import numpy as np
from enums import Datatype, Policy

DIV_NODE_CONF = 0.075
SAMPS_NEEDED = 15

class Agent:
  def __init__(self, name, environment, epsilon=0.05, policy=Policy.DEAF):
    self.name = name
    self.environment = environment
    self.reward_var = self.environment.reward_node
    self.epsilon = epsilon
    self.policy = policy
    self.friends = {}
    self.action_nodes = self.environment.action_nodes
    self.knowledge = Knowledge(self.environment)
    self.action_domains = gutil.only_given_keys(self.environment.domains, self.action_nodes)
    self.recent = None

  def act(self):
    givens = self.environment.pre.sample()
    choice = self.choose(givens)
    givens |= choice[1]
    env_act_feedback = self.environment.post.sample(givens)
    self.knowledge.add_obs(env_act_feedback) \
        if choice[0] == Datatype.OBS \
        else self.knowledge.add_exp(env_act_feedback)
    self.recent = env_act_feedback

  def choose(self, givens={}):
    if random.random() < self.epsilon:
      return (Datatype.EXP, self.experiment(givens))
    else:
      optimal_choice = self.optimal_choice(givens)
      return (Datatype.OBS, optimal_choice) if optimal_choice else (Datatype.EXP, self.experiment(givens))

  def experiment(self, givens={}):
    reward_vals = util.reward_vals(
      self.knowledge.get_useful_data(), self.action_nodes, self.reward_var, givens)
    unexplored = [util.dict_from_hash(e) for e in util.hashes_from_domain(self.action_domains) if e not in reward_vals.keys()]
    return random.choice(unexplored) if unexplored\
      else self.random_action()

  def optimal_choice(self, givens={}):
    my_data = self.knowledge.get_useful_data()
    
    if self.policy == Policy.NAIVE:
      for f in self.friends:
        my_data.extend(self.friends[f])
    
    if self.policy == Policy.SENSITIVE:
      friends_divs = self.divergences_from_friends()
      for f in friends_divs:
        if True not in friends_divs[f].values():
          my_data.extend(self.friends[f])
    
    expected_values = util.expected_vals(
        my_data, self.action_nodes, self.reward_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values\
      else self.random_action()

  def random_action(self):
    return random.choice(gutil.permutations(self.action_domains))

  def encounter(self, other):
    if self.policy == Policy.DEAF: return
    # friend_data = other.recent#other.knowledge.get_useful_data()
    if other.name not in self.friends:
      self.friends[other.name] = []
    if other.recent:
      self.friends[other.name].append(other.recent)

  def divergence_from_other(self, other_data):
    divergence = {}
    for node in self.knowledge.model.get_observable():
      if node in self.action_nodes: continue
      divergence[node] = self.knowledge.kl_divergence_of_node(node, other_data)
    return divergence

  def divergences_from_friends(self):
    divergences = {}
    for agent in self.friends:
      divergences[agent] = self.divergence_from_other(self.friends[agent])
    return divergences

  def divergent_nodes(self):
    is_divergent_dict = {}
    friend_divs = self.divergences_from_friends()
    for f in friend_divs:
      is_divergent_dict[f] = {}
      for node in friend_divs[f]:
        is_divergent_dict[f][node] = False \
          if friend_divs[f][node] != None \
          and abs(friend_divs[f][node]) < DIV_NODE_CONF \
          and len(self.friends[f]) >= SAMPS_NEEDED \
          else True
    return is_divergent_dict

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.name == other.name and self.environment == other.environment and self.reward_var == other.reward_var



if __name__ == "__main__":
  domains = {"W": (0, 1), "X": (0, 1), "Z": (0, 1), "Y": (0, 1)}
  environment = Environment(domains, {
      "W": lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
      "X": AssignmentModel(["W"], None),
      "Z": discrete_model(["X"], {(0,): [0.75, 0.25], (1,): [0.25, 0.75]}),
      "Y": discrete_model(["W", "Z"], {(0, 0): [1, 0], (0, 1): [1, 0], (1, 0): [1, 0], (1, 1): [0, 1]})
  })
  agent0 = Agent("zero", environment)
  agent1 = Agent("one", environment)
  # print(agent0.optimal_choice())
  # print(agent0.experiment())
  # print(agent0.knowledge.model.get_node_distributions())
  # print(agent0.knowledge.get_model_dist())
  for _ in range(1000):
    agent0.act()
    agent1.act()
    agent0.encounter(agent1)
    agent1.encounter(agent0)
  print(agent0.divergent_nodes())
  # print(agent0.reward({"Y": 1}))
  # model.draw_model()
  # print(agent0.knowledge.obs)
  # print(agent0.knowledge.get_conditional_prob({"Y": 1}, {"X": None}, agent0.knowledge.obs))
  # print(agent0.feature_vars)
  # print(agent0.knowledge.domains)
  # print(agent0.get_choice_combinations())

  # print(agent0.knowledge.model.get_distribution())
  # for p in agent0.knowledge.get_model_dist():
  #   print(p)

  # sample_query = "P(Y|X,W=1)"
  # print(sample_query)
  # print(util.parse_query(sample_query))
  # print()
  # for p in util.parse_query(sample_query):
  #   print(p)
  #   print(util.prob_with_unassigned(agent0.knowledge.domains, agent0.knowledge.obs, p[0], p[1]))
  #   print()
  
  # parsed_q = util.parse_query(sample_query)[0]
  # print(util.kl_divergence(agent0.knowledge.domains, agent0.knowledge.obs, agent0.knowledge.obs, parsed_q[0], parsed_q[1], 2))
