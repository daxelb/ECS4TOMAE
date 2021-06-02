# from causal_graph import CausalGraph
from knowledge import Knowledge
import util
import random
import math
from cgm import CausalGraph

class Agent:
  def __init__(self, name, model, domains, action_vars, reward_var):
    self.epsilon = 0.3
    self.name = name
    self.friends = {}

    self.knowledge = Knowledge(model, domains, action_vars)

    self.action_vars = action_vars
    self.reward_var = reward_var
    # assumptions: outcome_vars are all leaves in the model,
    # feature_vars are all parents of action_vars
    self.outcome_vars = self.knowledge.model.get_leaves()
    self.feature_vars = list()
    for node in self.action_vars:
      for parent in self.knowledge.model.get_parents(node):
        self.feature_vars.append(parent)
    self.action_domains = {}
    for act_var in self.action_vars:
      self.action_domains[act_var] = self.knowledge.domains[act_var]

  def choose(self, givens={}):
    if random.random() < self.epsilon:
      return self.experiment(givens)
    else:
      optimal_choice = self.optimal_choice(givens)
      return optimal_choice if optimal_choice else self.experiment(givens)

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

  def compare_distributions(self, other_dist):
    my_dist = self.knowledge.get_model_dist()


    # print(self.action_domains)
    # choices = util.permutations(self.action_domains)
    # print(choices)
    # random.shuffle(choices)
    # print(choices)
    # for choice in choices:
    #   print(choice)
      # if choice not in self.knowledge.obs and choice not in self.knowledge.exp:
        # print()


if __name__ == "__main__":
  edges = [("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")]
  model = CausalGraph(edges)
  domains = {"W": (0,1), "X": (0,1), "Y": (0,1), "Z": (0,1)}
  # # action_domains = {"X": [0,1]}
  agent0 = Agent("zero", model, domains, "X", "Y")
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([1,0,0,0])
  agent0.knowledge.add_obs([1,0,0,1])
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([0,1,0,0])
  # print(agent0.optimal_choice())
  print(agent0.experiment())
  print(model.get_node_distributions())
  print(agent0.knowledge.get_model_dist())
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
