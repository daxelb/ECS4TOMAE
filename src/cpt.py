from util import only_given_keys, permutations, hash_from_dict, Counter
from query import Query, Product, Summation
from re import findall
from math import inf

from numpy import random
from cgm import CausalGraph
class Knowledge:
  def __init__(self, rng, cgm, domains, act_var, rew_var):
    self.rng = rng
    self.cgm = cgm
    self.domains = domains
    self.act_var = act_var
    self.rew_var = rew_var
    self.cpts = {
      var: CPT(var, self.cgm.get_parents(var), domains) \
        for var in domains if var != X
    }
    self.rew_query = self.get_rew_query()
    
  def observe(self, sample):
    self.recent = sample
    for cpt in self.cpts.values():
      cpt.add(sample)
    
  def listen(self, sample):
    for cpt in self.cpts.values():
      cpt.add(sample)

  def get_rew_query(self):
    dist_vars = self.cgm.causal_path(self.act_var, self.rew_var)
    return Product(
      self.cgm.get_node_dist(v)
      for v in dist_vars
    ).assign(self.domains)
    
  def expected_rew(self, givens):
    query = self.rew_query.assign(givens)
    sum_over_rew = 0
    for rew_val in self.domains[self.rew_var]:
      query[self.rew_var] = rew_val
      if query.all_assigned():
        sum_over_rew += self.exp_rew_addition(rew_val, query)
        continue
      for q in query.over_unassigned():
        sum_over_rew += self.exp_rew_addition(rew_val, q)
    return sum_over_rew
  
  def exp_rew_addition(self, rew_val, query):
    try:
      return rew_val * query.solve(self.cpts)
    except TypeError:
      # is this the right thing to return when query.solve returns None?...
      return 0
    
  def optimal_choice(self, givens):
    best_choice = []
    best_rew = -inf
    choices = permutations({self.act_var: self.domains[self.act_var]})
    for choice in choices:
      expected_rew = self.expected_rew({**choice, **givens})
      if expected_rew is not None:
        if expected_rew > best_rew:
          best_choice = [choice]
          best_rew = expected_rew
        elif expected_rew == best_rew:
          best_choice.append(choice)
    return self.rng.choice(best_choice) if best_choice else None
  
  def all_causal_path_nodes_corrupted(self, agent):
    return self.cgm.causal_path(self.act_var, self.rew_var).issubset(set(self.div_nodes(agent)))
  
  def thompson_sample(self, givens):
    max_sample = 0
    choices = []
    for action in permutations(self.act_dom):
      alpha = 0
      beta = 0
      for agent in self.databank:
        if self.all_causal_path_nodes_corrupted(agent):
          continue
        rew_query = self.get_rew_query()
        rew_query.over()
        for w in (0,1):
          alpha_y_prob = self.solve_query(agent, Query({"Y": 1}, {**{"W": w}, **givens}))
          beta_y_prob = 1 - alpha_y_prob if alpha_y_prob is not None else None
          w_prob = self.solve_query(agent, Query({"W": w}, action))
          if alpha_y_prob is None or w_prob is None:
            continue
          else:
            count = self.databank[agent].num({**action, **givens})
            alpha += w_prob * alpha_y_prob * count
            beta += w_prob * beta_y_prob * count
      # alpha /= transport_agents
      # beta /= transport_agents
      sample = self.rng.beta(alpha + 1, beta + 1)
      if sample > max_sample:
        max_sample = sample
        choices = [action]
      if sample == max_sample:
        choices.append(action)
    return self.rng.choice(choices)
  

class CPT:
  def __init__(self, var, parents, domains):
    self.var = var
    self.parents = set(parents)
    self.domains = only_given_keys(domains, self.parents | {self.var})
    parent_assignments = permutations(only_given_keys(domains, parents))
    self.table = {hash_from_dict(pa): Counter() for pa in parent_assignments} if self.has_parents() else Counter()

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
      
  def as_query(self):
    return Query(self.var, self.parents)
      
  def size(self):
    return sum(sum(e.values()) for e in self.table.values())
  
  def is_empty(self):
    return self.size() == 0
  
  def has_parents(self):
    return bool(self.parents)
  
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
  # var = "Y"
  # parents = {"Z", "W"}
  # a = CPT(var, parents, domains)
  # print(a.size())
  # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # a.add({"W": 0, "X": 0, "Z": 1, "Y": 0})
  # print(a.prob(Query({"Y": 1}, {"W": 0, "Z": 1})))
  # print(a.size())
  rng = random.default_rng(100)
  nodes = ["W", "X", "Y", "Z"]
  edges = [("Z","X"),("Z", "Y"), ("X", "W"), ("W", "Y")]
  cgm = CausalGraph(nodes, edges, set_nodes={"X"})
  k = Knowledge(rng, cgm, domains, "X", "Y")
  # print(k.rew_query)
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 0})
  k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  print(k.expected_rew({"X": 0, "Z": 1}))
  print(k.optimal_choice({"Z": 1}))
  print(k.cpts["Y"].size())
