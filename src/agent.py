from model import CausalGraph

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
    return self.model.parents(node)

  def get_children(self, node):
    return self.model.get_children()

  def get_exogenous(self):
    return self.model.get_exogenous()

  def get_endogenous(self):
    return self.model.get_endogenous()

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
  def __init__(self, model):
    self.knowledge = Knowledge(model)


if __name__ == "__main__":
  model = CausalGraph([("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])
  agent0 = Agent(model)
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([1,1,0,1])
  agent0.knowledge.add_obs([1,0,0,0])
  agent0.knowledge.add_obs([0,0,0,0])
  agent0.knowledge.add_obs([0,1,1,1])
  model.draw_model()
  print(agent0.knowledge.observations)
  print(agent0.knowledge.get_prob(("Y", 1), {"X": 1}))
  # print(agent0.observations[agent0.observations["Y"].isin([0,1])])