from query import Query
import gutil
from collections.abc import MutableSequence, Iterable
from copy import deepcopy

def is_Q(obj):
  return isinstance(obj, (Query, Queries, Summation, Product, Quotient))

def is_num(obj):
  return isinstance(obj, (int, float))

class Queries(MutableSequence):
  def __init__(self, data=None):
    super(Queries, self).__init__()
    self._list = list() if data is None else list(data) if isinstance(data, Iterable) else [data]

  def remove_dupes(self):
    assert not isinstance(self, (Summation, Product))
    gutil.remove_dupes(self._list)
  
  def get_assignments(self):
    assignments = {}
    for q in self:
      assignments |= q.get_assignments()
    return assignments

  def get_unassigned(self):
    unassigned = {}
    for q in self.unpack():
      if isinstance(q, Query):
        unassigned.update(q.get_unassigned())
    return unassigned
  
  def contains(self, var):
    for q in self:
      if is_Q(q) and q.contains(var):
        return True
    return False
  
  def included_domains(self, domains):
    included = {}
    for var in domains:
      if self.contains(var):
        included[var] = domains[var]
    return included
  
  def over(self, domains):
    return self.over_helper(self.included_domains(domains), self)
  
  def over_unassigned(self):
    return self.over_helper(self.get_unassigned(), self)
  
  def over_helper(self, unassigned, assignments):
    if not unassigned:
      return assignments
    var = list(unassigned.keys())[0]
    dom = unassigned.pop(var)
    new_assignments = Queries()
    for a in dom:
      new_assignments.append(assignments.deepcopy().assign_one(var, a))
    return self.over_helper(unassigned, new_assignments)
  
  def assign(self, var_or_dict, ass=None):
    if ass is None:
      return self.assign_many(var_or_dict)
    else:
      return self.assign_one(var_or_dict, ass)
  
  def assign_one(self, var, ass):
    for q in self:
      if is_Q(q):
        q = q.assign_one(var, ass)
    return self
      
  def assign_many(self, ass_dict):
    for q in self:
      if is_Q(q):
        q = q.assign_many(ass_dict)
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
    return "<{}: {}>".format(self.__class__.__name__, str(self))
  
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
    joiner = " * " if isinstance(self, Product) else " + " if isinstance(self, Summation) else ", "
    for i, e in enumerate(self._list):
      s += str(e)
      s += joiner if i < len(self._list) - 1 else "]"
    return s
  
  def __copy__(self):
    return self.__class__((self._list))
  
  def deepcopy(self):
    return self.__class__(deepcopy(self._list))

  def insert(self, i, val):
    self._list.insert(i, val)
    
  def append(self, val):
    self.insert(len(self._list), val)
    
  def remove(self, val):
    self._list.remove(val)
  
  def __iter__(self):
    for e in self._list:
      yield e
  
  
class Summation(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      data = data._list
    super().__init__(data)
    for i, q in enumerate(self._list):
      if isinstance(q, Queries) and not isinstance(q, Product):
        self._list[i] = Summation(q)
    
  def solve(self, data):
    summation = 0
    for q in self._list:
      summation += q.solve(data) if is_Q(q) else q
    return summation

class Product(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      data = data._list
    super().__init__(data)
    for i, q in enumerate(self._list):
      if isinstance(q, Queries) and not isinstance(q, Summation):
        self._list[i] = Product(q)
    
  def solve(self, data):
    assert all(is_Q(q) or is_num(q) for q in self._list)
    product = 1
    for q in self._list:
      product *= q.solve(data) if is_Q(q) else q
    return product
  
class Quotient():
  def __init__(self, nume=None, denom=None):
    assert is_Q(nume) or is_num(nume)
    assert is_Q(denom) or is_num(denom)
    self.nume = nume
    self.denom = denom
    
  def solve(self, data):
    return self.nume.solve(data) if is_Q(self.nume) else self.nume \
        / self.denom.solve(data) if is_Q(self.denom) else self.denom
        
  def contains(self, var):
    return self.nume.contains(var) or self.denom.contains(var)
  
  def assign(self, dict_or_var, assignment=None):
    if assignment is None:
      return self.assign_many(dict_or_var)
    else:
      return self.assign_one(dict_or_var, assignment)
  
  def assign_one(self, var, ass):
    if is_Q(self.nume):
      self.nume.assign_one(var, ass)
    if is_Q(self.denom):
      self.denom.assign_one(var, ass)
    return self
    
  def assign_many(self, ass_dict):
    if is_Q(self.nume):
      self.nume.assign_many(ass_dict)
    if is_Q(self.denom):
      self.denom.assign_many(ass_dict)
    return self
    
  def __repr__(self):
    return "<Quotient: {}>".format(str(self))
    
  def __str__(self):
    return "({}) / ({})".format(str(self.nume), str(self.denom))
  
if __name__ == "__main__":
  q = Query({"Y": 1}, {"X": 0, "Z":1, "S":None})
  q1 = Query({"S": None}, {"Z": 1})
  qs = Product([q1, q])
  qs.assign_many({"S": (0,1)})
  # data = [{"X": 1, "Y": 0, "W": 1}]
  # print(qs.solve(data))
  so = Summation(qs.over_unassigned())
  quo = Quotient(qs, so)
  # print(quo)
  print(Summation(qs.over({"X": (0,1), "R": (0,1)})))
  # print(so)
  