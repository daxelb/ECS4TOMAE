import numpy as np
from util import permutations, Counter

def randomize(rng, iter):
  new_iter = []
  num_elements = len(iter)
  for i in range(num_elements):
    if i == num_elements - 1:
      new_iter.append(1-sum(new_iter))
      break
    new_iter.append(rng.uniform(0,1-sum(new_iter)))
  rng.shuffle(new_iter)
  return tuple(new_iter)
class RandomModel:
  def __init__(self, probs):
    self._probs = tuple(probs)
    self.domain = tuple(range(len(probs)))
    self.parents = tuple()

  def model(self, rng, **kwargs):
    return rng.choice(self.domain, p=self._probs)
  
  def randomize(self, rng, rand_prob=1.0):
    if rand_prob != 1.0 and rng.random() >= rand_prob:
      return self
    return RandomModel(randomize(rng, self._probs))
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and set(self.domain) == set(other.domain) \
        and self._probs == other._probs
        
  def __repr__(self):
    return "RandomModel(Domains: {}, Probabilities:{})".format(self.domain, self._probs)
  
  def __call__(self, *args, **kwargs):
    assert len(args) == 1
    return self.model(args[0], **kwargs)

class DiscreteModel:
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
    self.domain = tuple(range(len(list(lookup_table.values())[0])))

  def prob(self, assignments, my_assignment=None):
    assignment_domains = dict()
    assignment_probs = dict()
    for p in self.parents:
      assignment_domains[p] = tuple(assignments[p].keys()) if isinstance(assignments[p], dict) else (assignments[p],)
      assignment_probs[p] = tuple(assignments[p].values()) if isinstance(assignments[p], dict) else (1,)
    prob_dist = Counter()
    for a, probs in zip(permutations(assignment_domains), permutations(assignment_probs)):
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


  def model(self, rng, **kwargs):
    a = tuple([kwargs[p] for p in self.parents])
    b = None
    for m, p in zip(self._inputs, self._ps):
      if a == m:
        b = rng.choice(self._outputs, p=p)

    if b == None:
      raise ValueError(
          "It looks like an input was provided which doesn't have a lookup.")
    return int(b)
  
  def randomize(self, rng, rand_prob=1.0):
    if rand_prob != 1.0 and rng.random() >= rand_prob:
      return self
    new_table = dict(self.lookup_table)
    for input, probs in new_table.items():
      new_table[input] = randomize(rng, probs)
    return DiscreteModel(self.parents, new_table)
  
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
    return "DiscreteModel(Parents: {}, Probabilities: {})".format(self.parents, self._ps)
  
  def __call__(self, *args, **kwargs):
    assert len(args) == 1
    return self.model(args[0], **kwargs)


class ActionModel:
  def __init__(self, parents, domain):
    self.parents = parents
    self.domain = tuple(domain)
    
  def randomize(self, rng=None, rand_prob=1):
    return ActionModel(self.parents, self.domain)
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and set(self.domain) == set(other.domain) \
        and set(self.parents) == set(other.parents)
        
  def __repr__(self):
    return "ActionModel(Parents: {0}, Domains: {1})".format(self.parents, self.domain)
