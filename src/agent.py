# from causal_graph import CausalGraph
from knowledge import Knowledge
import util
import random
from environment import Environment
from cam import CausalAssignmentModel, discrete_model
import numpy as np

DIV_NODE_CONF = 0.08
SAMPS_NEEDED = 8

class Agent:
  def __init__(self, name, environment, reward_var):
    self.epsilon = 0.99
    self.name = name
    self.environment = environment
    self.reward_var = reward_var
    self.friends = {}
    self.action_vars = self.environment.action_nodes
    self.knowledge = Knowledge(self.environment.cgm, self.environment.domains, self.action_vars)
    self.action_domains = util.only_specified_keys(self.environment.domains, self.action_vars)

  def choose(self, givens={}):
    if random.random() < self.epsilon:
      return ("exp", self.experiment(givens))
    else:
      optimal_choice = self.optimal_choice(givens)
      return ("obs", optimal_choice) if optimal_choice else ("exp", self.experiment(givens))

  def experiment(self, givens={}):
    reward_vals = util.reward_vals(
      self.knowledge.get_useful_data(), self.action_vars, self.reward_var, givens)
    unexplored = [util.dict_from_hash(e) for e in util.hashes_from_domain(self.action_domains) if e not in reward_vals.keys()]
    return random.choice(unexplored) if unexplored\
      else self.random_action()

  def optimal_choice(self, givens={}):
    expected_values = util.expected_vals(
        self.knowledge.get_useful_data(), self.action_vars, self.reward_var, givens)
    return util.dict_from_hash(util.max_key(expected_values)) if expected_values\
      else self.random_action()

  def random_action(self):
    return util.random_assignment(self.action_domains)

  def act(self):
    givens = self.environment.pre.sample()
    choice = self.choose(givens)
    givens |= choice[1]
    dp = self.environment.post.sample(givens)
    if choice[0] == "obs":
      self.knowledge.add_obs(dp)
    else:
      self.knowledge.add_exp(dp)

  def encounter(self, other):
    friend_data = other.knowledge.get_useful_data()
    if other.name not in self.friends:
      self.friends[other.name] = []
    if friend_data:
      self.friends[other.name].append(friend_data[-1])

  def divergence(self, other_data):
    divergence = {}
    for node in self.knowledge.model.get_observable():
      if node in self.action_vars: continue
      divergence[node] = self.knowledge.kl_divergence_of_node(node, other_data)
    return divergence

  def divergences(self):
    divergences = {}
    for agent in self.friends:
      divergences[agent] = self.divergence(self.friends[agent])
    return divergences

  def divergent_nodes(self):
    is_divergent_dict = {}
    friend_divs = self.divergences()
    for f in friend_divs:
      is_divergent_dict[f] = {}
      for node in friend_divs[f]:
        is_divergent_dict[f][node] = False \
          if friend_divs[f][node] != None \
          and abs(friend_divs[f][node]) < DIV_NODE_CONF \
          and len(self.friends[f]) >= SAMPS_NEEDED \
          else True
    return is_divergent_dict



if __name__ == "__main__":
  domains = {"W": (0, 1), "X": (0, 1), "Z": (0, 1), "Y": (0, 1)}
  environment = Environment(domains, {
      "W": lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
      "X": CausalAssignmentModel(["W"], None),
      "Z": discrete_model(["X"], {(0,): [0.75, 0.25], (1,): [0.25, 0.75]}),
      "Y": discrete_model(["W", "Z"], {(0, 0): [1, 0], (0, 1): [1, 0], (1, 0): [1, 0], (1, 1): [0, 1]})
  })
  agent0 = Agent("zero", environment, "Y")
  agent1 = Agent("one", environment, "Y")
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
