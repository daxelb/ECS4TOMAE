from model import CausalGraph
import util

class Knowledge():
  def __init__(self, model):
    self.model = model
    self.var_names = self.model.get_observable()
    self.observations = self.experiments = {}
    for var_name in self.var_names:
      self.observations[var_name] = []
      self.experiments[var_name] = []

  def get_observable(self):
    return self.model.get_observable()

  def get_parents(self, node):
    return self.model.get_parents(node)

  def get_children(self, node):
    return self.model.get_children()

  def get_exogenous(self):
    return self.model.get_exogenous()

  def get_endogenous(self):
    return self.model.get_endogenous()
  
  def get_leaves(self):
    return self.model.get_leaves()

  def draw_model(self, v=False):
    self.model.draw().render('output/causal-model.gv', view=v)
  
  def add_obs(self, sample):
    if len(self.var_names) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(self.var_names, sample))
    for var_name in self.var_names:
      self.observations[var_name].append(sample[var_name])
  
  def add_exp(self, sample):
    if len(self.var_names) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(self.var_names, sample))
    for var_name in self.var_names:
      self.experiments[var_name].append(sample[var_name])

  def get_prob(self, Q, e = {}, datatype="obs"):
    dataset = self.observations if datatype == "obs" else self.experiments
    Q_and_e_count = e_count = 0
    for i in range(len(list(dataset.values())[0])):
      consistent = True
      for key in self.var_names:
        if key in e.keys() and dataset[key][i] != e[key]:
          consistent = False
          break
      if consistent:
        e_count += 1
        if dataset[Q[0]][i] == Q[1]:
          Q_and_e_count += 1
    return Q_and_e_count / e_count

class Agent:
  def __init__(self, model, action_domains):
    self.knowledge = Knowledge(model)
    self.action_domains = action_domains

    # assumptions: outcome_vars are all leaves in the model,
    # feature_vars are all parents of action_vars (determined action_domain)
    self.action_vars = self.action_domains.keys()
    self.outcome_vars = self.knowledge.get_leaves()
    self.feature_vars = list()
    for node in self.action_vars:
      for parent in self.knowledge.get_parents(node):
        self.feature_vars.append(parent)

  def get_choice_combinations(self):
    return util.get_combinations(self.action_domains)

if __name__ == "__main__":
  model = CausalGraph([("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])
  action_domains = {"X": [0,1]}
  agent0 = Agent(model, action_domains)
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([1,1,0,1])
  agent0.knowledge.add_obs([1,0,0,0])
  agent0.knowledge.add_obs([0,0,0,0])
  agent0.knowledge.add_obs([0,1,1,1])
  model.draw_model()
  # print(agent0.knowledge.observations)
  # print(agent0.knowledge.get_prob(("Y", 1), {"X": 1}))
  # print(agent0.feature_vars)
  # print(agent0.knowledge.domains)
  print(agent0.get_choice_combinations())
  # print(agent0.observations[agent0.observations["Y"].isin([0,1])])