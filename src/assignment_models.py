"""
Defines Assignment Model classes, which are used to describe the behaviors and
probability distributions of nodes in a Structural Causal Model

CREDIT:
Much of this file/class was written by Iain Barr (#ijmbarr on GitHub)
from his public repository, causalgraphicalmodels, which is registered with the MIT License.
The code has been imported and modified into this project for ease/consistency
"""

import numpy as np
from util import permutations, Counter


def randomize(rng, iter):
  new_iter = []
  num_elements = len(iter)
  for i in range(num_elements):
    if i == num_elements - 1:
      new_iter.append(1-sum(new_iter))
      break
    new_iter.append(rng.uniform(0, 1-sum(new_iter)))
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
    return RandomModel(randomize(rng, self._probs))

  def prob(self, assignment):
    assert assignment in self.domain
    probs = dict()
    for val in self.domain:
      probs[val] = 1 if val == assignment else 0
    return probs

  def __repr__(self):
    return "RandomModel(Domain: {}, Probs:{})".format(self.domain, self._probs)

  def __call__(self, *args, **kwargs):
    assert len(args) == 1
    return self.model(args[0], **kwargs)

  def __reduce__(self):
    return (self.__class__, (self._probs,))


class DiscreteModel:
  def __init__(self, parents, lookup_table):
    assert len(parents) > 0
    self._lookup_table = lookup_table
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
      assignment_domains[p] = tuple(assignments[p].keys()) if isinstance(
          assignments[p], dict) else (assignments[p],)
      assignment_probs[p] = tuple(assignments[p].values()) if isinstance(
          assignments[p], dict) else (1,)
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

  def randomize(self, rng):
    new_table = dict(self._lookup_table)
    for input, probs in new_table.items():
      new_table[input] = randomize(rng, probs)
    return DiscreteModel(self.parents, new_table)

  def __repr__(self):
    return "DiscreteModel(Parents: {}, Probs: {})".format(self.parents, self._ps)

  def __call__(self, *args, **kwargs):
    assert len(args) == 1
    return self.model(args[0], **kwargs)

  def __reduce__(self):
    return (self.__class__, (self.parents, self._lookup_table))


class ActionModel:
  def __init__(self, parents, domain):
    self.parents = parents if parents is not None else tuple()
    self.domain = tuple(domain)

  def randomize(self, rng=None, rand_prob=1.0):
    return self

  def prob(self, assignment):
    assert assignment in self.domain
    probs = dict()
    for val in self.domain:
      probs[val] = 1 if val == assignment else 0
    return probs

  def __repr__(self):
    return "ActionModel(Parents: {0}, Domain: {1})".format(self.parents, self.domain)

  def __reduce__(self):
    return (self.__class__, (self.parents, self.domain))
