import putil
import util
import gutil
from collections.abc import Iterable
class Query:
  def __init__(self, Q, e={}):
    self.Q = self.parse_entry(Q)
    self.e = self.parse_entry(e)

  def parse_entry(self, entry):
    if isinstance(entry, dict):
      return entry
    if isinstance(entry, str):
      entry = [entry]
    if isinstance(entry, Iterable):
      return {var: None for var in entry}
    return
  
  def Q_and_e(self):
    return {**self.Q, **self.e}
  
  def get_vars(self):
    return sorted(list(self.Q_and_e().keys()))
  
  def get_assignments(self):
    assignments = {}
    for var, ass in self.Q_and_e().items():
      if not isinstance(ass, Iterable) and ass is not None:
        assignments |= {var: ass}
    return assignments
  
  def get_unassigned_vars(self):
    unassigned = set()
    for var, ass in self.Q_and_e().items():
      if isinstance(ass, Iterable):
        unassigned.add(var)
    return sorted(unassigned)
  
  def get_unassigned(self):
    return gutil.only_given_keys(self.Q_and_e(), self.get_unassigned_vars())
  
  def contains(self, var):
    return var in self.Q_and_e()
  
  def all_assigned(self):
    return len(self.get_unassigned()) == 0
  
  def solve(self, data):
    return putil.prob(data, self.Q, self.e)
  
  def assign(self, var_or_dict, ass=None):
    if ass is None:
      return self.assign_many(var_or_dict)
    else:
      return self.assign_one(var_or_dict, ass)
  
  def assign_one(self, var, ass):
    if var in self.Q:
      self.Q[var] = ass
    if var in self.e:
      self.e[var] = ass
      
  def assign_many(self, domains):
    for var, ass in domains.items():
      self.assign_one(var, ass)
    return self
  
  def as_tup(self):
    return (self.Q, self.e)
  
  def __hash__(self):
    return hash((tuple(sorted(self.Q.items())),tuple(sorted(self.e.items()))))
  
  def __repr__(self):
    return "<Query: {}>".format(str(self))
  
  def __str__(self):
    if self.e:
      return "P({}|{})".format(util.hash_from_dict(self.Q), util.hash_from_dict(self.e))
    return "P({})".format(util.hash_from_dict(self.Q))
  
  def __eq__(self, other):
    for var in self.Q:
      if var not in other.Q or self.Q[var] != other.Q[var]:
        return False
    for var in self.e:
      if var not in other.e or self.e[var] != other.e[var]:
        return False
    return True