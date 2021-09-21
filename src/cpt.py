from util import only_given_keys, permutations, hash_from_dict, Counter
from query import Query
import re

class CPT:
  def __init__(self, var, parents, domains):
    self.var = var
    self.parents = set(parents)
    self.domains = only_given_keys(domains, self.parents | {self.var})
    parent_assignments = permutations(only_given_keys(domains, parents))
    self.table = {hash_from_dict(pa): Counter() for pa in parent_assignments}

  def add_obs(self, obs):
    givens = hash_from_dict(only_given_keys(obs, self.parents))
    assert givens in self.table
    self.table[givens][obs[self.var]] += 1
  
  def count(self, query):
    c = self[query]
    if isinstance(c, dict):
      return sum(c.values())
    return c
  
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
    header = sorted(list(self.parents)) + [self.var] + ['']
    format_row = '\n' + '{:<3}' * len(header)
    res += format_row.format(*header)
    res += format_row.format(*['-' * len(e) for e in header])
    for pa in self.table:
      for va in self.domains[self.var]:
        row = []
        for parent_val in re.findall(r'\d+', pa):
          row.append(parent_val)
        row += [str(va), str(self.table[pa][va])]
        res += format_row.format(*row)
    return res + '\n'
      
if __name__ == "__main__":
  domains = {"Y": (0,1), "X": (0,1), "W": (0,1), "Z": (0,1)}
  var = "Y"
  parents = {"Z", "W"}
  a = CPT(var, parents, domains)
  print(a)
  obs1 = {"W": 0, "X": 0, "Z": 1, "Y": 1}
  a.add_obs(obs1)
  print(a)
  print(a.count({"W": 0, "Z": 1}))
  print(a.count({"W": 0, "Z": 0, "Y": 0}))
