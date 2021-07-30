from typing import List
from pandas import DataFrame
import pandas as pd
import gutil
import util
import math
from copy import copy
import numpy as np

def pairs(lst):
  return [(a, b) for i, a in enumerate(lst) for b in lst[i + 1:]]

class DataSet(list):
  def __init__(self, data=[]):
    super().__init__(data)

  def is_empty(self):
    return len(self) == 0
    
  def get_recent(self):
    return self[-1]
    
  def query(self, query_dict):
    res = DataSet()
    for e in self:
      consistent = True
      for key in query_dict:
        if e[key] != query_dict[key]:
          consistent = False
          break
      if consistent:
        res.append(e)
    return res
    
  def mean(self, var):
    total = len(self)
    return sum([e[var] for e in self]) / total if total else None
  
  def optimal_choice(self, act_dom, rew_var, givens):
    best_choice = None
    best_rew = -math.inf
    for choice in gutil.permutations(act_dom):
      expected_rew = self.query({**choice, **givens}).mean(rew_var)
      if expected_rew is not None and expected_rew > best_rew:
        best_choice = choice
        best_rew = expected_rew
    return best_choice



class DataBank:
  def __init__(self, domains, act_var, rew_var, data={}):
    self.data = data
    self.domains = domains
    self.vars = set(domains.keys())
    self.act_var = act_var
    self.rew_var = rew_var
    self.divergence = {}
    for key in self.data:
      self.add_agent(key)
  
  def add_agent(self, new_agent):
    # new_agent = hash(new_agent)
    if new_agent in self.data:
      return
    self.data[new_agent] = DataSet()
    self.divergence[new_agent] = {}
    for existing_agent in self.data.keys():
      self.divergence[new_agent][existing_agent] = {}
      self.divergence[existing_agent][new_agent] = {}
      for node in self.get_non_act_nodes():
        self.divergence[existing_agent][new_agent][node] = 1
        self.divergence[new_agent][existing_agent][node] = 1
        
  def get_non_act_nodes(self):
    return [node for node in self.vars if node != self.act_var]
        
  def kl_div_of_query(self, query, P_agent, Q_agent):
    return util.kl_divergence(self.domains, self.data[P_agent], self.data[hash(Q_agent)], query)
  
  def kl_div_of_node(self, node, P_agent, Q_agent):
    return self.kl_div_of_query(P_agent.knowledge.model.get_node_dist(node), P_agent, Q_agent)
        
  def update_divergence(self):
    for P_agent, P_data in self.data.items():
      if len(P_data) < P_agent.samps_needed:
          break
      for Q_agent, Q_data in self.data.items():
        if P_agent == Q_agent:
          continue
        for node in self.get_non_act_nodes():
          query = P_agent.knowledge.model.get_node_dist(node)
          self.divergence[P_agent][Q_agent][node] = util.kl_divergence(self.domains, P_data, Q_data, query)
    return
  
  def div_nodes(self, P_agent, Q_agent):
    if P_agent == Q_agent:
      return []
    return [node for node, divergence in self.divergence[P_agent][Q_agent].items() if divergence is None or abs(divergence) > P_agent.div_node_conf]
  
  def is_divergent_dict(self, P_agent, Q_agent):
    divergent = {}
    for node in self.get_non_act_nodes():
      node_divergence = self.divergence[P_agent][Q_agent][node]
      divergent[node] = node_divergence is None or \
                        abs(node_divergence) > P_agent.div_node_conf
    return divergent

  def all_data(self):
    data = DataSet()
    [data.extend(d) for d in self.data.values()]
    return data
  
  def sensitive_data(self, P_agent):
    data = DataSet()
    [data.extend(Q_data) for Q_agent, Q_data in self.data.items() if not self.div_nodes(P_agent, Q_agent)]
    return data

  def items(self):
    return self.data.items()

  def __getitem__(self, key):
    return self.data[key]

  def __getstate__(self):
    return self.__dict__
  
  def __reduce__(self):
    return type(self), (self.domains, self.act_var, self.rew_var, self.data)

    
if __name__ == "__main__":
  from agent import Agent
  from environment import Environment
  from enums import Policy
  from assignment_models import ActionModel, DiscreteModel, RandomModel
  baseline = Environment({
    "W": RandomModel((0.4, 0.6)),
    "X": ActionModel(("W"), (0, 1)),
    "Z": DiscreteModel(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": DiscreteModel(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  })
  db = DataBank(baseline.domains, baseline.act_var, baseline.rew_var)
  agents = [
    Agent("00", baseline, db, Policy.DEAF, 0.05, 0.03, 10),
    Agent("01", baseline, db, Policy.ADJUST, 0.05, 0.03, 10),
  ]
  
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[0], baseline.post.sample(set_values={"W": 0, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 1}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 1, "X": 0}))
  db.append(agents[1], baseline.post.sample(set_values={"W": 0, "X": 0}))

  from query import Query
  print(db.all_data().optimal_choice(agents[0].environment.get_act_dom(), "Y", {"W": 1}))