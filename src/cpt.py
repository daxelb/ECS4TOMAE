from util import only_given_keys, permutations, hash_from_dict, Counter
from query import Query, Product
from re import findall
from math import inf

from numpy import random
from cgm import CausalGraph
class Knowledge:
  def __init__(self, rng, cgm, domains, act_var, rew_var):
    self.cgm = cgm
    self.domains = domains
    self.act_var = act_var
    self.rew_var = rew_var
    self.cpts = {
      var: CPT(var, self.cgm.get_parents(var), domains) \
        for var in domains if var != act_var
    }
    self.rew_query = self.get_rew_query()
    
  def observe(self, observation):
    for cpt in self.cpts.values():
      cpt.add(observation)
    
  def get_rew_query(self):
    dist_vars = self.cgm.an(self.rew_var).difference(self.cgm.an(self.act_var))
    return Product([
      self.cgm.get_node_dist(v)
      for v in dist_vars
    ])
    
  def expected_rew(self, givens):
    summation = 0
    for rew_val in self.domains[self.rew_var]:
      product = 1
      for query in self.rew_query:
        product *= rew_val * self.cpts[query.query_var()].prob({**givens, **{self.rew_var: rew_val}})
      summation += product
    return summation
    
  def optimal_choice(self, givens):
    best_choice = []
    best_rew = -inf
    for choice in permutations(self.domains[self.act_var]):
      expected_rew = self.expected_rew({**choice, **givens})
      if expected_rew is not None:
        if expected_rew > best_rew:
          best_choice = [choice]
          best_rew = expected_rew
        elif expected_rew == best_rew:
          best_choice.append(choice)
    return self.rng.choice(best_choice) if best_choice else None

class CPT:
  def __init__(self, var, parents, domains):
    self.var = var
    self.parents = set(parents)
    self.domains = only_given_keys(domains, self.parents | {self.var})
    parent_assignments = permutations(only_given_keys(domains, parents))
    self.table = {hash_from_dict(pa): Counter() for pa in parent_assignments} if self.has_parents() else Counter()

  def has_parents(self):
    return bool(self.parents)

  def add(self, obs):
    if self.has_parents():
      givens = hash_from_dict(only_given_keys(obs, self.parents))
      assert givens in self.table
      self.table[givens][obs[self.var]] += 1
    else:
      self.table[obs[self.var]] += 1
  
  def count(self, query):
    c = self[query]
    if isinstance(c, dict):
      return sum(c.values())
    return c
  
  def prob(self, query):
      try:
        if isinstance(query, Query):
          return self.count(query) / self.count(query.e)
        return self.count(query) / self.count(only_given_keys(query, self.parents))
      except ZeroDivisionError:
        return None
      
  def size(self):
    return sum([sum(e.values()) for e in a.table.values()])
  
  def is_empty(self):
    return self.size() == 0
  
  def __getitem__(self, key):
    if isinstance(key, Query):
      if isinstance(key.Q[self.var], int):
        return self.table[hash_from_dict(key.e)][key.Q[self.var]]
      return self.table[hash_from_dict(key.e)]
    if isinstance(key, dict):
      if self.var in key:
        return self.table[hash_from_dict(only_given_keys(key, self.parents))][key[self.var]]
      return self.table[hash_from_dict(only_given_keys(key, self.parents))]
    return self.table[key]
  
  def __str__(self):
    res = ''
    header = sorted(list(self.parents)) + [self.var, '']
    format_row = '\n' + '{:<3}' * len(header)
    res += format_row.format(*header)
    res += format_row.format(*['-' * len(e) for e in header])
    for pa in self.table:
      for va in self.domains[self.var]:
        row = []
        for parent_val in findall(r'\d+', pa):
          row.append(parent_val)
        row += [str(va), str(self.table[pa][va])]
        res += format_row.format(*row)
    return res + '\n'
      
if __name__ == "__main__":
  domains = {"Y": (0,1), "X": (0,1), "W": (0,1), "Z": (0,1)}
  var = "Y"
  parents = {"Z", "W"}
  a = CPT(var, parents, domains)
  print(a.size())
  a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  a.add({"W": 0, "X": 0, "Z": 1, "Y": 0})
  print(a.prob(Query({"Y": 1}, {"W": 0, "Z": 1})))
  print(a.size())
  rng = random.default_rng(100)
  nodes = ["W", "X", "Y", "Z"]
  edges = [("Z","X"),("Z", "Y"), ("X", "W"), ("W", "Y")]
  cgm = CausalGraph(nodes, edges, set_nodes={"X"})
  k = Knowledge(rng, cgm, domains, "X", "Y")
  print(k.rew_query)
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 0})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  print(k.expected_rew({"X": 0, "Z": 1, "W": 0}))
