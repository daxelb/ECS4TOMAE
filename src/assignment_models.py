import numpy as np
import gutil
from copy import copy
class AssignmentModel:
    """
    Basically just a hack to allow me to provide information about the
    arguments of a dynamically generated function.
    """
    def __init__(self, parents, domain):
        self.parents = parents
        self.domain = domain

    def __call__(self, *args, **kwargs):
        assert len(args) == 0
        return self.model(**kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "(Parents: {0}, Domains: {1})".format(self.parents, self.domain)
    
    # def __copy__(self):
    #   return self.__class__(copy(self.parents), copy(self.name))

class RandomModel(AssignmentModel):
  def __init__(self, probs):
    self._probs = probs
    self.domain = list(range(len(probs)))
    self.parents = tuple()

  def model(self, **kwargs):
    return np.random.choice(self.domain, p=self._probs)
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and set(self.domain) == set(other.domain) \
        and self._probs == other._probs
        
  def __repr__(self):
    return self.__class__.__name__ + "(Parents: {0}, Domains: {1}, Probabilities:{2})".format(self.parents, self.domain, self._probs)
  
  # def __copy__(self):
  #   return self.__class__(copy(self._probs))
    
class DiscreteModel(AssignmentModel):
  def __init__(self, parents, lookup_table):
    assert len(parents) > 0
    self.lookup_table = lookup_table
    self.parents = parents

    # create input/output mapping
    self._inputs, weights = zip(*lookup_table.items())

    output_length = len(weights[0])
    assert all(len(w) == output_length for w in weights)
    self._outputs = np.arange(output_length)
    self._ps = [np.array(w) / sum(w) for w in weights]
    self.domain = list(range(len(gutil.first_value(lookup_table))))

  def prob(self, assignments, my_assignment=None):
    assignment_domains = dict()
    assignment_probs = dict()
    for p in self.parents:
      assignment_domains[p] = tuple(assignments[p].keys()) if isinstance(assignments[p], dict) else (assignments[p],)
      assignment_probs[p] = tuple(assignments[p].values()) if isinstance(assignments[p], dict) else (1,)
    prob_dist = gutil.Counter()
    for a, probs in zip(gutil.permutations(assignment_domains), gutil.permutations(assignment_probs)):
      for key, value in self.prob_helper(a, multiplier=np.prod(list(probs.values()))).items():
        prob_dist[key] += value
    return prob_dist if my_assignment is None else prob_dist

  def prob_helper(self, assignments, my_assignment=None, multiplier=1):
    a = tuple([assignments[p] for p in self.parents])
    b = None
    for m, p in zip(self._inputs, self._ps):
      if a == m:
        b = dict(zip(self._outputs, p * multiplier))
    if b == None:
      raise ValueError(
          "It looks like an input was provided which doesn't have a lookup.")
    return b if my_assignment is None else b[my_assignment]


  def model(self, **kwargs):
    a = tuple([kwargs[p] for p in self.parents])
    b = None
    for m, p in zip(self._inputs, self._ps):
      if a == m:
        b = np.random.choice(self._outputs, p=p)

    if b == None:
      raise ValueError(
          "It looks like an input was provided which doesn't have a lookup.")
    return int(b)
  
  def __eq__(self, other):
    if len(self._ps) != len(other._ps):
      return False
    for i in range(len(self._ps)):
      for j in range(len(self._ps[i])):
        if self._ps[i][j] != other._ps[i][j]:
          return False
    return isinstance(other, self.__class__) \
        and (set(self.domain) == set(other.domain)) \
        and (set(self.parents) == set(other.parents)) \
        and (set(self._inputs) == set(other._inputs)) \
        and (set(self._outputs) == set(other._outputs))
        
  def __repr__(self):
    return self.__class__.__name__ + "(Parents: {0}, Probabilities: {1})".format(self.parents, self._ps)
  
  # def __copy__(self):
  #   return self.__class__(copy(self.parents), copy(self.lookup_table))

class ActionModel(AssignmentModel):
  def __init__(self, parents, domain):
    super().__init__(parents, domain)

  def model(self, **kwargs):
    return None
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and set(self.domain) == set(other.domain) \
        and set(self.parents) == set(other.parents)
        
  def __repr__(self):
    return self.__class__.__name__ + "(Parents: {0}, Domains: {1})".format(self.parents, self.domain)
