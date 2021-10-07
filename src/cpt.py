from copy import deepcopy
from util import only_given_keys, permutations, hash_from_dict, Counter
from query import Query, Product, Summation, Count
from re import findall
from math import inf

from numpy import random
from cgm import CausalGraph  

class CPT:
  def __init__(self, var, parents, domains):
    self.var = var
    self.parents = set(parents)
    self.domains = only_given_keys(domains, self.parents | {self.var})
    self.query = Count(self.var, self.parents)
    self.table = {key: 0 for key in Count(self.var, self.parents).unassigned_combos(self.domains)}

  def add(self, obs):
    self.table[self.query.assign(obs)] += 1

  def update(self, other):
    for key in self.table:
      self.table[key] += other.table[key]
  
  def __getitem__(self, key):
    if isinstance(key, Query):
      if key in self.table:
        return self.table[key]
      else:
        summ = 0
        for k in self.table:
          if k.issubset(key):
            summ += self.table[k]
        return summ
      
  def __deepcopy__(self, memo):
    dc = self.__class__(self.var, self.parents, self.domains)
    dc.table = deepcopy(self.table)
    return dc
  
  def __str__(self):
    res = ''
    header = sorted(list(self.parents)) + [self.var, str(self.query)]
    format_row = '\n' + '{:<3}' * len(header)
    res += format_row.format(*header)
    res += format_row.format(*['-' * len(e) for e in header])
    for key in self.table:
      row = []
      for var_val in findall(r'\d+', str(key)):
        row.append(var_val)
      row.append(str(self.table[key]))
      res += format_row.format(*row)
    return res + '\n'
      
if __name__ == "__main__":
  cpt_y = CPT("Y", ("W", "Z"), {"Y": (0,1), "W": (0,1), "Z": (0,1)})
  cpt_w = CPT("W", "X", {"W": (0,1), "X": (0,1)})
  cpt_w.add({"X": 0, "W": 0})
  cpt_w.add({"X": 0, "W": 0})
  cpt_w.add({"X": 0, "W": 0})
  cpt_w.add({"X": 0, "W": 1})
  cpt_w.add({"X": 1, "W": 1})
  cpt_w.add({"X": 1, "W": 1})
  cpt_w.add({"X": 1, "W": 1})
  cpt_w.add({"X": 1, "W": 0})
            
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 0, "Z": 0, "W": 0})
  cpt_y.add({"Y": 1, "Z": 0, "W": 0})
  #
  cpt_y.add({"Y": 1, "Z": 0, "W": 1})
  cpt_y.add({"Y": 1, "Z": 0, "W": 1})
  cpt_y.add({"Y": 1, "Z": 0, "W": 1})
  cpt_y.add({"Y": 1, "Z": 0, "W": 1})
  cpt_y.add({"Y": 1, "Z": 0, "W": 1})
  cpt_y.add({"Y": 0, "Z": 0, "W": 1})
  cpt_y.add({"Y": 0, "Z": 0, "W": 1})
  cpt_y.add({"Y": 0, "Z": 0, "W": 1})
  cpt_y.add({"Y": 0, "Z": 0, "W": 1})
  cpt_y.add({"Y": 0, "Z": 0, "W": 1})
  #
  cpt_y.add({"Y": 1, "Z": 1, "W": 0})#
  cpt_y.add({"Y": 1, "Z": 1, "W": 0})
  cpt_y.add({"Y": 1, "Z": 1, "W": 0})
  cpt_y.add({"Y": 1, "Z": 1, "W": 0})
  cpt_y.add({"Y": 1, "Z": 1, "W": 0})
  cpt_y.add({"Y": 0, "Z": 1, "W": 0})
  cpt_y.add({"Y": 0, "Z": 1, "W": 0})
  cpt_y.add({"Y": 0, "Z": 1, "W": 0})
  cpt_y.add({"Y": 0, "Z": 1, "W": 0})
  cpt_y.add({"Y": 0, "Z": 1, "W": 0})
#
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})#
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 1, "Z": 1, "W": 1})
  cpt_y.add({"Y": 0, "Z": 1, "W": 1})
  cpts = {"Y": cpt_y, "W": cpt_w}
  rew_query = Product([Query({"Y": 1},{"Z": 1, "W": (0,1)}), Query({"W": (0,1)},{"X": 0})])
  # print(rew_query)
  rew_query = Summation(rew_query.over())
  
  # print(cpts["Y"][Count({"Y": 1},{"Z": 1, "W": 1})])
  # print(cpts["Y"][Count({"Y": (0,1)},{"Z": 1, "W": 1})])
  # print(cpts["W"][Count({"W": (0,1)},{"X": 1})])
  # print(cpts["W"][Count({"W": (0,1)},{"X": 0})])
  # print(Count({"Y": 1},{"Z": 1, "W": 1}).solve(cpts["Y"]))
  # print(Count({"Y": 0},{"Z": 1, "W": 1}).solve(cpts["Y"]))
  # print(Count({"Y": (0,1)},{"Z": 1, "W": 1}).solve(cpts["Y"]))
  # print(Query({"Y": 1},{"Z": 1, "W": 1}).solve(cpts["Y"]))
  print(rew_query.solve(cpts))
  
  
  
  
  
  
  
  # domains = {"Y": (0,1), "X": (0,1), "W": (0,1), "Z": (0,1)}
  # # var = "Y"
  # # parents = {"Z", "W"}
  # # a = CPT(var, parents, domains)
  # # print(a.size())
  # # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # # a.add({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # # a.add({"W": 0, "X": 0, "Z": 1, "Y": 0})
  # # print(a.prob(Query({"Y": 1}, {"W": 0, "Z": 1})))
  # # print(a.size())
  # rng = random.default_rng(100)
  # nodes = ["W", "X", "Y", "Z"]
  # edges = [("Z","X"),("Z", "Y"), ("X", "W"), ("W", "Y")]
  # cgm = CausalGraph(nodes, edges, set_nodes={"X"})
  # k = Knowledge(rng, cgm, domains, "X", "Y")
  # # print(k.rew_query)
  # k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # k.observe({"W": 0, "X": 0, "Z": 1, "Y": 0})
  # k.observe({"W": 0, "X": 0, "Z": 1, "Y": 1})
  # print(k.expected_rew({"X": 0, "Z": 1}))
  # print(k.optimal_choice({"Z": 1}))
  # print(k.cpts["Y"].size())
