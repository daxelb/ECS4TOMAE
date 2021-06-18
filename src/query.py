import util
import gutil
from collections.abc import MutableSequence
from copy import deepcopy

class Queries(MutableSequence):
  def __init__(self, data=None):
    super(Queries, self).__init__()
    self._list = list() if data is None else list(data)

  def remove_dupes(self):
    gutil.remove_dupes(self._list)
    
  def sum_over_unassigned(self, domains):
    domains = gutil.only_given_keys(domains, )
    return Summation(permutations)
  
  def get_unassigned(self):
    unassigned = set()
    for q in self._list:
      if isinstance(q, Query):
        unassigned.update(q.get_unassigned())
    return unassigned
  
  def over_unassigned(self):
    unassigned = self.get_unassigned()
    self.over_unassigned_helper(unassigned, self.copy())
  
  def over_unassigned_helper(self, unassigned, assignments):
    var_to_assign = unassigned.pop()
    for a in assignments:
      var_domain = 
      new_a = Queries([a])
      a = new_a
    unassigned = self.get_unassigned()
    
  def assignment_options(self, domain):
    return
    
  def __repr__(self):
    return "<{0} {1}>".format(self.__class__.__name__, self._list)
  
  def __len__(self):
    return len(self._list)
  
  def __getitem__(self, i):
    return self._list[i]
  
  def __delitem__(self, i):
    del self._list[i]
    
  def __setitem__(self, i, val):
    self._list[i] = val
    
  def __str__(self):
    return str(self._list)
  
  def __copy__(self):
    return Queries(self._list)
  
  def __deepcopy__(self):
    return Queries(deepcopy(self._list))

  def insert(self, i, val):
    self._list.insert(i, val)
    
  def append(self, val):
    self.insert(len(self._list), val)
    
  def remove(self, val):
    self._list.remove(val)
  
  def __iter__(self):
    for e in self._list:
      yield e
  
class Product(Queries):
  def __init__(self, data=None):
    super().__init__(data)
    
  def solve(self, data):
    assert all(isinstance(q, (Query, int, float)) for q in self._list)
    product = 1
    for q in self._list:
      product *= q.solve(data) if isinstance(q, Query) else q
    return product

class Summation(Queries):
  def __init__(self, data=None):
    super().__init__(data)
    
  def solve(self, data):
    assert all(isinstance(q, (Query, int, float)) for q in self._list)
    summation = 0
    for q in self._list:
      summation += q.solve(data) if isinstance(q, Query) else q
    return summation
  

class Query:
  def __init__(self, Q, e={}):
    self.Q = Q
    self.e = e
    self.Qe = {**Q, **e}
  
  def get_assignments(self):
    assignments = {}
    for var, ass in self.Qe.items():
      if ass is not None:
        assignments |= {var: ass}
    return assignments
  
  def get_unassigned(self):
    unassigned = set()
    for var, ass in self.Qe.items():
      if ass is None:
        unassigned.add(var)
    return unassigned
  
  def solve(self, data):
    return util.prob(data, self.Q, self.e)
  
  def as_tup(self):
    return (self.Q, self.e)
  
  def __hash__(self):
    return hash((tuple(sorted(self.Q.items())),tuple(sorted(self.e.items()))))
  
  def __repr__(self):
    return "<Q={}, e={}>".format(self.Q, self.e)
  
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
        
  
if __name__ == "__main__":
  q = Query({"Y": 1}, {"X": None, "W":1})
  q1 = Query({"Y": 1}, {"X": None, "Z": None})
  # queries = {q, q1}
  # print(queries)
  # print(q.as_tup())
  # print(q.get_assignments())
  # print(q.get_unassigned())
  # print(q.__eq__(q1))
  # print(q)
  # print(q1)
  # print(q.__hash__())
  qs = Summation([q, q1])
  # print(list(qs.__iter__()))
  # data = [{"X": 1, "Y": 0, "W": 1}]
  # print(qs.solve(data))
  print(qs.get_unassigned())
  