import numpy as np
import gutil

class AssignmentModel:
    """
    Basically just a hack to allow me to provide information about the
    arguments of a dynamically generated function.
    """
    def __init__(self, parents, domain):
        self.parents = parents
        # self.model = model
        self.domain = domain

    def __call__(self, *args, **kwargs):
        assert len(args) == 0
        return self.model(**kwargs)

    def __repr__(self):
        return "AssignmentModel({})".format(",".join(self.parents))

    def __eq__(self, other):
        return isinstance(other, AssignmentModel) and self.parents == other.parents and self.model == other.model and self.domain == other.domain

class RandomModel(AssignmentModel):
  def __init__(self, probs):
    self._probs = probs
    self.domain = list(range(len(probs)))
    self.parents = []

  def model(self, **kwargs):
    return np.random.choice(self.domain, p=self._probs)
    
class DiscreteModel(AssignmentModel):
  def __init__(self, parents, lookup_table):
    assert len(parents) > 0
    self.parents = parents

    # create input/output mapping
    self._inputs, weights = zip(*lookup_table.items())

    output_length = len(weights[0])
    assert all(len(w) == output_length for w in weights)
    self._outputs = np.arange(output_length)
    self._ps = [np.array(w) / sum(w) for w in weights]
    self.domain = list(range(len(gutil.first_value(lookup_table))))

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

class ActionModel(AssignmentModel):
  def __init__(self, parents, domain):
    self.parents = parents
    self.domain = domain

  def model(self, **kwargs):
    return None