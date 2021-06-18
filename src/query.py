import putil
import util
import gutil
from collections.abc import MutableSequence, Iterable
import copy

class Queries(MutableSequence):
  def __init__(self, data=None):
    super(Queries, self).__init__()
    self._list = list() if data is None else list(data)

  def remove_dupes(self):
    gutil.remove_dupes(self._list)
  
  def get_unassigned(self):
    unassigned = {}
    for q in self.unpack():
      if isinstance(q, Query):
        unassigned.update(q.get_unassigned())
    return unassigned
  
  def over_unassigned(self):
    return self.over_unassigned_helper(self.get_unassigned(), self)
  
  def over_unassigned_helper(self, unassigned, assignments):
    if not unassigned:
      return assignments
    var = list(unassigned.keys())[0]
    dom = unassigned.pop(var)
    new_assignments = Queries()
    for a in dom:
      new_assignments.append(assignments.deepcopy().assign(var, a))
    return self.over_unassigned_helper(unassigned, new_assignments)
  
  def assign(self, var, ass):
    for q in self:
      if isinstance(q, Queries):
        q = q.assign(var, ass)
      elif isinstance(q, Query):
        if var in q.Q:
          q.Q[var] = ass
        if var in q.e:
          q.e[var] = ass
    return self
  
  def unpack(self):
    new_self = self.__copy__()
    for i in range(len(self._list)):
      q = new_self._list[i]
      if isinstance(q, Queries):
        new_self._list.pop(i)
        q = q.unpack()
        for j in range(len(q)):
          new_self._list.insert(i + j, q[j])
        return new_self.unpack()
    return new_self
    
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
    s = "["
    for i, e in enumerate(self._list):
      s += str(e)
      s += ", " if i < len(self._list) - 1 else "]"
    if isinstance(self, (Product, Summation)):
      return "<{}: {}>".format(self.__class__.__name__, s)
    return "<{}>".format(s)
  
  def __copy__(self):
    return self.__class__((self._list))
  
  def deepcopy(self):
    return self.__class__((copy.deepcopy(self._list)))

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
    if isinstance(data, Queries):
      super().__init__(data._list)
    else:
      super().__init__(data)
    
  def solve(self, data):
    assert all(isinstance(q, (Product, Summation, Query, int, float)) for q in self._list)
    product = 1
    for q in self._list:
      product *= q if isinstance(q, int, float) else q.solve(data)
    return product

class Summation(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      super().__init__(data._list)
    else:
      super().__init__(data)
    
  def solve(self, data):
    summation = 0
    for q in self._list:
      summation += q if isinstance(q, int, float) else q.solve(data)
    return summation

class Query:
  def __init__(self, Q, e={}):
    self.Q = Q
    self.e = e
    self.Qe = {**Q, **e}
  
  def get_assignments(self):
    assignments = {}
    for var, ass in self.Qe.items():
      if not isinstance(ass, Iterable):
        assignments |= {var: ass}
    return assignments
  
  def get_unassigned_vars(self):
    unassigned = set()
    for var, ass in self.Qe.items():
      if isinstance(ass, Iterable):
        unassigned.add(var)
    return sorted(unassigned)
  
  def get_unassigned(self):
    return gutil.only_given_keys(self.Qe, self.get_unassigned_vars())
  
  def solve(self, data):
    return putil.prob(data, self.Q, self.e)
  
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
  q = Query({"Y": 1}, {"X": (0,1), "W":1})
  q1 = Query({"Y": 1}, {"X": (0,1), "Z": (0,1)})
  # queries = {q, q1}
  # print(queries)
  # print(q.as_tup())
  # print(q.get_assignments())
  # print(q.get_unassigned())
  # print(q.__eq__(q1))
  # print(q)
  # print(q1)
  # print(q.__hash__())
  qs = Product([q1, q])
  # print(q.get_unassigned())
  # print(list(qs.__iter__()))
  # data = [{"X": 1, "Y": 0, "W": 1}]
  # print(qs.solve(data))
  # print(qs.get_unassigned())
  so = Summation(qs.over_unassigned())
  # print(so)
  for q in so:
    print(q)
  # print(qs.over_unassigned())
  # print(qs.unpack())
  