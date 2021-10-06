from networkx.classes.function import create_empty_copy
from data import DataSet
from util import hash_from_dict, only_given_keys, permutations
from collections.abc import MutableSequence, Iterable
from copy import deepcopy, copy


def is_Q(obj):
  return isinstance(obj, (Query, Queries))


def is_num(obj):
  return isinstance(obj, (int, float))


class Query(object):
  def __init__(self, Q, e={}):
    self.Q = self.parse_entry(Q)
    self.e = self.parse_entry(e)
    self.q_e = self.Q_and_e()

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

  def solve(self, data):
    cpt = data[self.var()] if isinstance(data, dict) else data
    count_e = Count(self.e)
    count_e = count_e.solve(cpt)
    if count_e == 0:
      return None
    return Count(self).solve(cpt) / count_e

  def var(self):
    """
    Previously called "var()"
    Renamed for simplicity
    Returns the variable of the query var if there is only one
    Will return an assertion error if now
    Ex:
    q = P(Y|W,Z)
    q.var() = "Y"
    """
    assert len(self.Q) == 1
    return list(self.Q.keys())[0]

  def get_unassigned(self):
    return {var: ass for var, ass in self.items() if isinstance(ass, Iterable) or ass is None}

  def all_assigned(self):
    return len(self.get_unassigned()) == 0

  def combos(self, domains):
    if self.all_assigned():
      return {self}
    return {copy(self).assign(combo) for combo in permutations(domains)}

  def unassigned_combos(self, domains):
    if self.all_assigned():
      return {self}
    unassigned = permutations(only_given_keys(domains, self.get_unassigned()))
    return {copy(self).assign(combo) for combo in unassigned}

  def parse_as_df_query(self):
    str_query = ""
    for var, ass in self.Q_and_e().items():
      if is_num(ass):
        str_query += "{0} == {1} & ".format(var, ass)
    return str_query[:-3]

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

  def assign_many(self, assignments):
    for var, ass in assignments.items():
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

  def items(self):
    return self.Q_and_e().items()

  def __copy__(self):
    return self.__class__(copy(self.Q), copy(self.e))

  def __hash__(self):
    return hash((
        tuple(sorted(self.Q.items())),
        tuple(sorted(self.e.items()))
    ))

  def __str__(self):
    if self.e:
      return "P({}|{})".format(hash_from_dict(self.Q), hash_from_dict(self.e))
    return "P({})".format(hash_from_dict(self.Q))

  def __eq__(self, other):
    return all(var in other.Q for var in self.Q) \
        and all(self.Q[var] == other.Q[var] for var in self.Q) \
        and all(var in other.e for var in self.e) \
        and all(self.e[var] == other.e[var] for var in self.e)

  def __bool__(self):
    return bool(self.Q_and_e())

  def __len__(self):
    return len(self.Q_and_e())

  def __setitem__(self, key, val):
    return self.assign_one(key, val)

  def __getitem__(self, key):
    return self.Q_and_e()[key]

  def __contains__(self, item):
    if isinstance(item, Iterable):
      return all(e in self.Q_and_e().keys() for e in item)
    return item in self.Q_and_e()


class Count(Query):
  def __init__(self, Q, e={}):
    if isinstance(Q, Query):
      super(self.__class__, self).__init__({**Q.Q, **Q.e})
    else:
      super(self.__class__, self).__init__(
          {**self.parse_entry(Q), **self.parse_entry(e)})

  def assign(self, var_or_dict, ass=None):
    self.assign_many(var_or_dict) if ass is None else self.assign_one(var_or_dict, ass)
    return self

  def assign_one(self, var, ass):
    if var in self.Q:
      self.Q[var] = ass
    if var in self.e:
      self.e[var] = ass
    # return self

  def assign_many(self, assignments):
    for var, ass in assignments.items():
      self.assign_one(var, ass)
    # return self

  def solve(self, cpt):
    return cpt[self]

  def issubset(self, other):
    assert isinstance(other, Count)
    return other.Q.items() <= self.Q.items()

  def __str__(self):
    return "N({})".format(hash_from_dict(self.Q))


class Queries(MutableSequence):
  def __init__(self, data=None):
    super(Queries, self).__init__()
    self._list = list() if data is None else list(
        data) if isinstance(data, Iterable) else [data]

  def get_vars(self):
    vars = set()
    for q in self:
      vars |= q.get_vars()
    return vars

  def get_unassigned(self):
    """
    Returns a all the unassigned variables
    of all queries in the object, returning
    a dictionary of their (domain/None) mappings
    """
    unassigned = dict()
    (unassigned.update(q.get_unassigned()) for q in self)
    return unassigned

  def all_assigned(self):
    return len(self.get_unassigned()) == 0

  def over(self, domains={}):
    self.assign(domains)
    if self.all_assigned():
      return self.__class__(self)
    assignments = permutations(self.get_unassigned())
    assigned = self.__class__()
    for ass in assignments:
      assigned.append(deepcopy(self).assign(ass))
    return assigned

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

  def __setitem__(self, key, val):
    if isinstance(key, int):
      self._list[key] = val
    else:
      for e in self._list:
        e[key] = val

  def __str__(self):
    return "[{}]".format(", ".join([str(e) for e in self._list]))

  def __copy__(self):
    return self.__class__(self._list)

  def __contains__(self, item):
    if isinstance(item, Iterable):
      for var in item:
        if not any(var in q for q in self):
          return False
      return True
    return any(item in q for q in self)

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

  def solve(self, cpts):
    assert self.all_assigned()
    summation = 0
    for q in self:
      try:
        if isinstance(q, (Summation, Product)):
          summation += q.solve(cpts)
        elif isinstance(q, Query):
          summation += q.solve(cpts[q.var()])
        else:
          summation += q
      except TypeError:
        return None
    return summation

  def __str__(self):
    if not len(self):
      return '0'
    ' + '.join([str(e) for e in self._list])


class Product(Queries):
  def __init__(self, data=None):
    if isinstance(data, Queries):
      data = data._list
    super().__init__(data)
    for i, q in enumerate(self._list):
      if isinstance(q, Queries) and not isinstance(q, Summation):
        self._list[i] = Product(q)

  def solve(self, cpts):
    if not self.all_assigned():
      return Summation(self.over()).solve(cpts)
    product = 1
    for q in self:
      try:
        if isinstance(q, (Summation, Product)):
          product *= q.solve(cpts)
        elif isinstance(q, Query):
          product *= q.solve(cpts[q.var()])
        else:
          product *= q
      except TypeError:
        return None
    return product

  def __str__(self):
    if not len(self):
      return "1"
    return ' * '.join([str(e) for e in self._list])
