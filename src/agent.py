from model import CausalGraph, SCM
import pandas as pd
import numpy as np

class Agent:
  def __init__(self, model):
    self.model = model
    self.var_names = self.model.get_variables()
    self.observations = self.experiments = {}
    for var_name in self.var_names:
      self.observations[var_name] = list()
      self.experiments[var_name] = list()

  def convert_to_dataframe(self, lst):
    return pd.DataFrame(lst, columns=self.model.get_variables())


  def add_sample(self, sample, datatype="obs"):
    if len(self.var_names) != len(sample):
      print("Error adding sample to agent's dataset.")
    dataset = self.observations if datatype == "obs" else self.experiments
    # Format the sample as a dictionary if it is passed in as a list
    # (assumes values are in alphabetical order)
    if type(sample) is list:
      sample = dict(zip(self.var_names, sample))
    for var_name in self.var_names:
      dataset[var_name].append(sample[var_name])

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

if __name__ == "__main__":
  model = CausalGraph([("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])
  agent0 = Agent(model)
  agent0.add_sample([0,1,1,1])
  agent0.add_sample([1,1,0,1])
  agent0.add_sample([1,0,0,0])
  agent0.add_sample([0,0,0,0])
  agent0.add_sample([0,1,1,1])
  model.draw_model()
  print(agent0.observations)
  print(agent0.get_prob(("Y", 1), {"X": 1}))
  # print(agent0.observations[agent0.observations["Y"].isin([0,1])])