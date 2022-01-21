"""
Defines the CPT (Conditional Probability Table) class. Which is how agents
store observations about connecting to the Causal Graphical model of their environment.
These are used to compute probabilities.
"""

from copy import deepcopy
from util import only_given_keys
from query import Query, Count
from re import findall


class CPT:
  def __init__(self, var, parents, domains):
    self.var = var
    self.parents = set(parents)
    self.domains = only_given_keys(domains, self.parents | {self.var})
    self.query = Count(self.var, self.parents)
    self.table = {key: 0 for key in Count(
        self.var, self.parents).unassigned_combos(self.domains)}

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
