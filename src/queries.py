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
  
  def over(self, domains):
    return self.over_helper(domains, self)
  
  def over_unassigned(self):
    return self.over_helper(self.get_unassigned(), self)
  
  def over_helper(self, unassigned, assignments):
    if not unassigned:
      return assignments
    var = list(unassigned.keys())[0]
    dom = unassigned.pop(var)
    new_assignments = Queries()
    for a in dom:
      new_assignments.append(assignments.deepcopy().assign(var, a))
    return self.over_helper(unassigned, new_assignments)
  
  def assign(self, var, ass):
    for q in self:
      if is_Q(q):
        q = q.assign(var, ass)
    return self
      
  def apply_domains(self, domains):
    for q in self:
      if is_Q(q):
        q = q.apply_domains(domains)
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
    new = self._list.copy()
    for q in new:
      if isinstance(q, Queries) and not isinstance(q, Product):
        q = Summation(q)
    self._list = new
    
  def solve(self, data):
    summation = 0
    for q in self._list:
      summation += q if isinstance(q, int, float) else q.solve(data)
    return summation

class Product(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      data = data._list
    super().__init__(data)
    
  def solve(self, data):
    assert all(isinstance(q, (Product, Summation, Query, int, float)) for q in self._list)
    product = 1
    for q in self._list:
      product *= q if isinstance(q, int, float) else q.solve(data)
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
  
  def assign(self, var, ass):
    if is_Q(self.nume):
      self.nume.assign(var, ass)
    if is_Q(self.denom):
      self.denom.assign(var, ass)
    return self
    
  def apply_domains(self, domains):
    if is_Q(self.nume):
      self.nume.apply_domains(domains)
    if is_Q(self.denom):
      self.denom.apply_domains(domains)
    return self
    
  def __repr__(self):
    return "<Quotient: {}>".format(str(self))
    
  def __str__(self):
    return "({}) / ({})".format(str(self.nume), str(self.denom))
  
if __name__ == "__main__":
  q = Query({"Y": 1}, {"X": 0, "Z":1, "S":None})
  q1 = Query({"S": None}, {"Z": 1})
  qs = Product([q1, q])
  qs.apply_domains({"S": (0,1)})
  # data = [{"X": 1, "Y": 0, "W": 1}]
  # print(qs.solve(data))
  so = Summation(qs.over_unassigned())
  quo = Quotient(qs, so)
  # print(quo)
  print(Summation(qs.over({"X": (0,1)})))
  # print(so)
  