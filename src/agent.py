from model import CausalGraph
import util
import random

class Knowledge():
  def __init__(self, model, domains):
    self.model = model
    self.domains = domains
    self.most_recent = None
    self.observations = self.experiments = {}
    for var_name in self.get_observable():
      self.observations[var_name] = []
      self.experiments[var_name] = []

  def get_observable(self):
    return self.model.get_observable()

  def get_parents(self, node):
    return self.model.get_parents(node)

  def get_children(self, node):
    return self.model.get_children(node)

  def get_exogenous(self):
    return self.model.get_exogenous()

  def get_endogenous(self):
    return self.model.get_endogenous()
  
  def get_leaves(self):
    return self.model.get_leaves()

  def draw_model(self, v=False):
    self.model.draw().render('output/causal-model.gv', view=v)
  
  def add_obs(self, sample):
    observable_vars = self.get_observable()
    if len(observable_vars) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(observable_vars, sample))
    for var_name in observable_vars:
      self.observations[var_name].append(sample[var_name])
    self.most_recent = sample
  
  def add_exp(self, sample):
    observable_vars = self.get_observable()
    if len(observable_vars) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(observable_vars, sample))
    for var_name in observable_vars:
      self.experiments[var_name].append(sample[var_name])
    self.most_recent = sample

  def format_conditional_distribution(self):
    dist = []
    for p in self.model.get_distribution().split("P")[1:]:
      new_p = ["", {}]
      query = True
      e_word = ""
      for char in p:
        if char == '|':
          query = False
        elif char != '(':
          if query:
            if char == ')':
              break
            else:
              new_p[0] += char
          else:
            if char == ',' or char == ')':
              # Using None to denote an unspecified value
              new_p[1][e_word] = None
              e_word = ""
            else:
              e_word += char
      dist.append(tuple(new_p))
    return dist
    # for eqn in self.model.get_distribution().split("P"):


class Agent:
  def __init__(self, model, domains, action_vars):
    self.knowledge = Knowledge(model, domains)

    self.action_vars = action_vars
    # assumptions: outcome_vars are all leaves in the model,
    # feature_vars are all parents of action_vars
    self.outcome_vars = self.knowledge.get_leaves()
    self.feature_vars = list()
    for node in self.action_vars:
      for parent in self.knowledge.get_parents(node):
        self.feature_vars.append(parent)


  def choose_optimally(self):
    action_domains = {}
    for act_var in self.action_vars:
      action_domains[act_var] = self.knowledge.domains[act_var]
    for choice in random.shuffle(util.combinations(action_domains)):
      if choice not in self.knowledge.observations and choice not in self.knowledge.experiments:
        print()


if __name__ == "__main__":
  model = CausalGraph([("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])
  domains = {"W": (0,1), "X": (0,1), "Y": (0,1), "Z": (0,1)}
  # action_domains = {"X": [0,1]}
  agent0 = Agent(model, domains, "X")
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([1,1,0,1])
  agent0.knowledge.add_obs([1,0,0,0])
  agent0.knowledge.add_obs([0,0,0,0])
  agent0.knowledge.add_obs([0,1,1,1])
  model.draw_model()
  # print(agent0.knowledge.observations)
  # print(agent0.knowledge.observations)
  # print(agent0.knowledge.get_conditional_prob({"Y": 1}, {"X": None}, agent0.knowledge.observations))
  # print(agent0.feature_vars)
  # print(agent0.knowledge.domains)
  # print(agent0.get_choice_combinations())
  print(util.conditional_prob({"Y": 1}, {"X": None}, agent0.knowledge.observations))
  print(util.prob({"Y": None}, agent0.knowledge.observations))
  # print(agent0.knowledge.format_conditional_distribution())
