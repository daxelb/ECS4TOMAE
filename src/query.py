import util
import gutil
from collections.abc import MutableSequence, Iterable
from copy import deepcopy, copy

def is_Q(obj):
  return isinstance(obj, (Query, Queries, Quotient))

def is_num(obj):
  return isinstance(obj, (int, float))
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
    return set(self.Q_and_e().keys())
  
  def query_var(self):
    assert len(self.Q) == 1
    return list(self.Q.keys())[0]
  
  def get_assignments(self, dictionary=None):
    if dictionary is None:
      dictionary = self.Q_and_e()
    assignments = {}
    [assignments.update({var: ass}) for var, ass in dictionary.items() if self.is_assigned(ass)]
    return assignments
  
  def is_assigned(self, assignment):
    return not isinstance(assignment, Iterable) and assignment is not None
  
  def get_unassigned_vars(self):
    unassigned = set()
    for var, ass in self.Q_and_e().items():
      if isinstance(ass, Iterable) or ass is None:
        unassigned.add(var)
    return unassigned
  
  def get_unassigned(self):
    return gutil.only_given_keys(self.Q_and_e(), self.get_unassigned_vars())
  
  def all_assigned(self):
    return len(self.get_unassigned()) == 0
  
  def contains_var(self, var):
    return var in self.Q_and_e()
  
  def contains(self, el):
    if isinstance(el, Iterable):
      for var in el:
        if self.contains_var(var):
          return True
      return False
    return self.contains_var(el)
  
  def combos(self, domains):
    if self.all_assigned():
      return {self}
    return {copy(self).assign(combo) for combo in gutil.permutations(domains)}
  
  def unassigned_combos(self, domains):
    if self.all_assigned():
      return {self}
    return {copy(self).assign_unassigned(combo) for combo in gutil.permutations(domains)}
  
  def parse_as_df_query(self):
    str_query = ""
    for var, ass in self.Q_and_e().items():
      if is_num(ass):
        str_query += "{0} == {1} & ".format(var,ass)
    return str_query[:-3]

  def num_consistent(self, data):
    return len(data.query(self.parse_as_df_query())) if self else len(data)

  def uncomputed_prob(self, data):
    # assert self.all_assigned()
    return (Query(self.Q_and_e()).num_consistent(data), Query(self.e).num_consistent(data))
  
  def solve(self, data):
    # assert self.all_assigned()
    query_space = data.query(self.e)
    return len(query_space.query(self.Q)) / len(query_space) if query_space else None
  
  def solve_unassigned(self, data, domains):
    return {q: q.solve(data) for q in self.unassigned_combos(domains)}
    
  def assign(self, var_or_dict, ass=None):
    return self.assign_many(var_or_dict) if ass is None else self.assign_one(var_or_dict, ass)
  
  def assign_one(self, var, ass):
    if var in self.Q:
      self.Q[var] = ass
    if var in self.e:
      self.e[var] = ass
    return self
      
  def assign_many(self, domains):
    for var, ass in domains.items():
      self.assign_one(var, ass)
    return self
  
  def assign_unassigned(self, var_or_dict, ass=None):
    return self.assign_unassigned_many(var_or_dict) if ass is None else self.assign_unassigned_one(var_or_dict, ass)
  
  def assign_unassigned_one(self, var, ass):
    if var in self.Q and self.Q[var] == None:
      self.Q[var] = ass
    if var in self.e and self.e[var] == None:
      self.e[var] = ass
    return self
  
  def assign_unassigned_many(self, domains):
    for var, ass in domains.items():
      self.assign_unassigned_one(var, ass)
    return self
  
  def as_tup(self):
    return (self.Q, self.e)
  
  def __copy__(self):
    return Query(copy(self.Q), copy(self.e))
  
  def __hash__(self):
    return hash((tuple(sorted(self.Q.items())),tuple(sorted(self.e.items()))))
  
  def __repr__(self):
    return "<Query: {}>".format(str(self))
  
  def __str__(self):
    if self.e:
      return "P({}|{})".format(util.hash_from_dict(self.Q), util.hash_from_dict(self.e))
    return "P({})".format(util.hash_from_dict(self.Q))
  
  def __eq__(self, other):
    return all(var in other.Q and self.Q[var] == other.Q[var] for var in self.Q) \
       and all(var in other.e and self.e[var] == other.e[var] for var in self.e)
  
  def __bool__(self):
    return len(self.Q_and_e()) != 0
  
class Queries(MutableSequence):
  def __init__(self, data=None):
    super(Queries, self).__init__()
    self._list = list() if data is None else list(data) if isinstance(data, Iterable) else [data]
    
  def get_vars(self):
    vars = set()
    for q in self:
      vars |= q.get_vars()
    return vars
    
  def Q(self):
    Q = dict()
    [Q.update(q.Q) for q in self]
    return Q
    
  def e(self):
    e = dict()
    [e.update(q.e) for q in self]
    return e
    
  def Q_and_e(self):
    Q_and_e = dict()
    [Q_and_e.update(q.Q_and_e()) for q in self]
    return Q_and_e
    
  def get_assignments(self, dictionary=None):
    if dictionary is None:
      dictionary = self.Q_and_e()
    assignments = {}
    [assignments.update(q.get_assignments(dictionary)) for q in self]
    return assignments

  def get_unassigned(self):
    unassigned = {}
    for q in self:
      unassigned.update(q.get_unassigned())
    return unassigned
  
  def all_assigned(self):
    return len(self.get_unassigned()) == 0
  
  def contains_var(self, var):
    for q in self:
      if is_Q(q) and q.contains(var):
        return True
    return False
  
  def contains(self, el):
    if isinstance(el, Iterable):
      for var in el:
        if self.contains(var):
          return True
      return False
    return self.contains_var(el)
  
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
      new_assignments.append(deepcopy(assignments).assign_one(var, a))
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
    new_self = copy(self)
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
    return "[{}]".format(", ".join([str(e) for e in self._list]))
  
  def __copy__(self):
    return self.__class__(self._list)
  
  # try to get this to work with __deepcopy__
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
      new_add = q.solve(data) if is_Q(q) else q
      if new_add is None:
        return None
      summation += new_add
    return summation
  
  def uncomputed_prob(self, data):
    summation = [0,0]
    for q in self._list:
      new_add = q.uncomputed_prob(data) if is_Q(q) else q
      if new_add is None:
        return None
      summation[0] += new_add[0]
      summation[1] += new_add[1]
    return summation
  
  def __str__(self):
    if not len(self):
      return "0"
    return "[{}]".format(" + ".join([str(e) for e in self._list]))

class Product(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      data = data._list
    super().__init__(data)
    for i, q in enumerate(self._list):
      if isinstance(q, Queries) and not isinstance(q, Summation):
        self._list[i] = Product(q)
    
  def solve(self, data):
    # assert all(is_Q(q) or is_num(q) for q in self._list)
    product = 1
    for q in self._list:
      new_mult = q.solve(data) if is_Q(q) else q
      if new_mult is None:
        return None
      product *= new_mult
    return product
  
  def uncomputed_prob(self, data):
    product = [0,0]
    for q in self._list:
      new_mult = q.uncomputed_prob(data) if is_Q(q) else q
      if new_mult is None:
        return None
      product[0] *= new_mult[0]
      product[1] *= new_mult[1]
    return product
    
  def __str__(self):
    if not len(self):
      return "1"
    return "[{}]".format(" * ".join([str(e) for e in self._list]))
  
class Quotient():
  def __init__(self, nume=None, denom=None):
    # assert is_Q(nume) or is_num(nume)
    # assert is_Q(denom) or is_num(denom)
    self.nume = nume
    self.denom = denom
    
  def solve(self, data):
    nume = self.nume.solve(data) if is_Q(self.nume) else self.nume
    denom = self.denom.solve(data) if is_Q(self.denom) else self.denom
    if nume is None or denom is None:
      return None
    return nume / denom
        
  def contains(self, el):
    return self.nume.contains(el) or self.denom.contains(el)
  
  def assign(self, dict_or_var, assignment=None):
    if assignment is None:
      return self.assign_many(dict_or_var)
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
  